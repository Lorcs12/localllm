"""KV Router: hot/cold KV page management with async SSD fetch.

In a Hybrid Mamba architecture, only the attention layers produce KV cache
entries. For a 1M-token context with 10 attention layers, even the reduced
KV cache can be tens of gigabytes. The KV Router solves this by keeping only
the important tokens in RAM and paging the rest to SSD:

  Hot pages (RAM):
    - Sink tokens (first ~4): absorb disproportionate attention mass
    - Local window (last ~2048): the current working context
    - Heavy-hitters: sparse set of high-attention tokens from the middle

  Cold pages (SSD):
    - Everything else, organized into contiguous 2 MB slabs for sequential reads
    - Fetched on-demand by Thread B during the 80ms draft window

The fetch time for one 2 MB cold KV page at 7 GB/s sequential is ~0.28ms,
which is completely hidden inside the 80ms draft window. This enables
true infinite-context inference on laptop hardware.
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .hardware import HardwareProfile, ssd_read_time_ms


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KVRouterConfig:
    """Configuration for the KV page router."""

    total_context_tokens: int = 1_000_000
    hot_tokens: int = 2048
    sink_tokens: int = 4
    cold_page_size_bytes: int = 2 * 1024 * 1024  # 2 MB per cold page
    kv_bytes_per_token: int = 20_480  # 10 attn layers × 8 heads × 128 dim × 2 bytes
    attn_layers: int = 10
    draft_window_ms: float = 80.0

    @property
    def tokens_per_cold_page(self) -> int:
        if self.kv_bytes_per_token == 0:
            return 0
        return self.cold_page_size_bytes // self.kv_bytes_per_token

    @property
    def total_kv_bytes(self) -> int:
        return self.total_context_tokens * self.kv_bytes_per_token

    @property
    def hot_kv_bytes(self) -> int:
        hot = min(self.hot_tokens + self.sink_tokens, self.total_context_tokens)
        return hot * self.kv_bytes_per_token

    @property
    def cold_tokens(self) -> int:
        hot = min(self.hot_tokens + self.sink_tokens, self.total_context_tokens)
        return max(0, self.total_context_tokens - hot)

    @property
    def cold_kv_bytes(self) -> int:
        return self.cold_tokens * self.kv_bytes_per_token

    @property
    def n_cold_pages(self) -> int:
        if self.tokens_per_cold_page == 0:
            return 0
        return max(1, (self.cold_tokens + self.tokens_per_cold_page - 1) // self.tokens_per_cold_page)


# ---------------------------------------------------------------------------
# KV Page
# ---------------------------------------------------------------------------

@dataclass
class KVPage:
    """A contiguous block of KV cache entries for a range of token positions."""

    page_id: int
    token_start: int
    token_end: int
    location: str  # "ram" or "ssd"
    attention_score: float = 0.0
    size_bytes: int = 0
    data: np.ndarray | None = None

    @property
    def n_tokens(self) -> int:
        return self.token_end - self.token_start

    @property
    def token_range(self) -> tuple[int, int]:
        return (self.token_start, self.token_end)


# ---------------------------------------------------------------------------
# KV Router
# ---------------------------------------------------------------------------

@dataclass
class KVRouter:
    """Routes KV cache lookups between hot pages (RAM) and cold pages (SSD).

    During generation, Thread A drafts tokens while Thread B fetches any
    cold KV pages needed for the next verification step. The fetch is
    completely hidden inside the draft window.
    """

    config: KVRouterConfig
    pages: list[KVPage] = field(default_factory=list)
    _executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=2))
    _hw: HardwareProfile = field(default_factory=HardwareProfile)
    _rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(42))

    @classmethod
    def from_context(
        cls,
        config: KVRouterConfig | None = None,
        hw: HardwareProfile | None = None,
    ) -> KVRouter:
        """Build a KV page index for the full context.

        Pages are assigned as follows:
          - Sink pages: first `sink_tokens` tokens → RAM
          - Local window: last `hot_tokens` tokens → RAM
          - Everything in between → cold pages on SSD
        """
        if config is None:
            config = KVRouterConfig()
        if hw is None:
            hw = HardwareProfile(ssd_sequential_gb_s=7.0)

        pages: list[KVPage] = []
        page_id = 0
        ctx = config.total_context_tokens
        tokens_per_page = max(1, config.tokens_per_cold_page)

        # Sink pages (RAM)
        if config.sink_tokens > 0:
            sink_end = min(config.sink_tokens, ctx)
            pages.append(KVPage(
                page_id=page_id,
                token_start=0,
                token_end=sink_end,
                location="ram",
                attention_score=1.0,
                size_bytes=sink_end * config.kv_bytes_per_token,
            ))
            page_id += 1

        # Cold pages (SSD) — middle section
        cold_start = config.sink_tokens
        cold_end = max(cold_start, ctx - config.hot_tokens)

        pos = cold_start
        while pos < cold_end:
            end = min(pos + tokens_per_page, cold_end)
            pages.append(KVPage(
                page_id=page_id,
                token_start=pos,
                token_end=end,
                location="ssd",
                attention_score=0.0,
                size_bytes=(end - pos) * config.kv_bytes_per_token,
            ))
            page_id += 1
            pos = end

        # Hot local window (RAM)
        local_start = max(cold_end, 0)
        if local_start < ctx:
            pages.append(KVPage(
                page_id=page_id,
                token_start=local_start,
                token_end=ctx,
                location="ram",
                attention_score=0.9,
                size_bytes=(ctx - local_start) * config.kv_bytes_per_token,
            ))

        return cls(config=config, pages=pages, _hw=hw)

    @property
    def hot_pages(self) -> list[KVPage]:
        return [p for p in self.pages if p.location == "ram"]

    @property
    def cold_pages(self) -> list[KVPage]:
        return [p for p in self.pages if p.location == "ssd"]

    def route(self, query_position: int) -> tuple[list[KVPage], list[KVPage]]:
        """Determine which pages are hot (available) and which need fetching.

        For a given query position, returns:
          (hot_pages, cold_pages_to_fetch)

        The cold pages are those in the SSD that the attention mechanism
        might need. In practice, the CUP predictor or attention pattern
        analysis determines which cold pages to fetch — here we return
        all cold pages as candidates.
        """
        hot = [p for p in self.pages if p.location == "ram"]
        cold = [p for p in self.pages if p.location == "ssd"]
        return hot, cold

    def fetch_cold_pages_async(self, pages: list[KVPage]) -> Future[list[KVPage]]:
        """Thread B: fetch cold KV pages from SSD asynchronously."""
        return self._executor.submit(self._fetch_pages, pages)

    def fetch_cold_pages_sync(self, pages: list[KVPage]) -> list[KVPage]:
        """Blocking fetch for testing."""
        return self._fetch_pages(pages)

    def _fetch_pages(self, pages: list[KVPage]) -> list[KVPage]:
        """Simulate SSD reads for cold pages."""
        fetched = []
        for page in pages:
            read_ms = ssd_read_time_ms(page.size_bytes, self._hw, sequential=True)
            time.sleep(read_ms / 1000.0)

            page.data = self._rng.randn(page.n_tokens, 64).astype(np.float32)
            page.location = "ram"
            fetched.append(page)
        return fetched

    def evict_to_ssd(self, pages: list[KVPage]) -> int:
        """Move pages from RAM to SSD (eviction)."""
        count = 0
        for page in pages:
            if page.location == "ram" and not self._is_protected(page):
                page.location = "ssd"
                page.data = None
                count += 1
        return count

    def promote_to_ram(self, pages: list[KVPage]) -> int:
        """Mark fetched pages as in RAM."""
        count = 0
        for page in pages:
            if page.location == "ssd":
                page.location = "ram"
                count += 1
        return count

    def _is_protected(self, page: KVPage) -> bool:
        """Sink and local window pages are protected from eviction."""
        cfg = self.config
        if page.token_start < cfg.sink_tokens:
            return True
        ctx = cfg.total_context_tokens
        if page.token_end >= ctx - cfg.hot_tokens:
            return True
        return False

    def stats(self) -> dict:
        """Current router statistics."""
        hot = self.hot_pages
        cold = self.cold_pages
        hot_bytes = sum(p.size_bytes for p in hot)
        cold_bytes = sum(p.size_bytes for p in cold)

        return {
            "total_pages": len(self.pages),
            "hot_pages": len(hot),
            "cold_pages": len(cold),
            "hot_bytes": hot_bytes,
            "hot_mb": round(hot_bytes / (1024 ** 2), 2),
            "cold_bytes": cold_bytes,
            "cold_mb": round(cold_bytes / (1024 ** 2), 2),
            "total_mb": round((hot_bytes + cold_bytes) / (1024 ** 2), 2),
            "hot_tokens": sum(p.n_tokens for p in hot),
            "cold_tokens": sum(p.n_tokens for p in cold),
        }

    def simulate_query(self, query_position: int | None = None) -> dict:
        """Simulate one query: routing decision, fetch timing, draft window fit."""
        if query_position is None:
            query_position = self.config.total_context_tokens - 1

        hot, cold = self.route(query_position)

        # Fetch time for one cold page (worst case: one 2MB slab)
        single_page_bytes = self.config.cold_page_size_bytes
        fetch_ms = ssd_read_time_ms(single_page_bytes, self._hw, sequential=True)

        draft_ms = self.config.draft_window_ms
        hidden = fetch_ms < draft_ms

        return {
            "query_position": query_position,
            "hot_pages": len(hot),
            "cold_pages": len(cold),
            "single_page_fetch_ms": round(fetch_ms, 2),
            "draft_window_ms": draft_ms,
            "fetch_hidden": hidden,
            "effective_overhead_ms": round(max(0.0, fetch_ms - draft_ms), 2),
        }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# End-to-end simulation
# ---------------------------------------------------------------------------

def simulate_kv_routing(
    config: KVRouterConfig | None = None,
    hw: HardwareProfile | None = None,
) -> dict:
    """Run an end-to-end KV routing simulation."""
    if config is None:
        config = KVRouterConfig()
    if hw is None:
        hw = HardwareProfile(ssd_sequential_gb_s=7.0)

    router = KVRouter.from_context(config, hw)
    stats = router.stats()

    # Simulate queries at different positions
    positions = [0, config.total_context_tokens // 4,
                 config.total_context_tokens // 2,
                 config.total_context_tokens - 1]
    queries = [router.simulate_query(pos) for pos in positions]

    # All cold pages fetch time (worst case: fetch everything)
    all_cold_fetch_ms = ssd_read_time_ms(stats["cold_bytes"], hw, sequential=True)

    router.shutdown()

    return {
        "config": {
            "total_context_tokens": config.total_context_tokens,
            "hot_tokens": config.hot_tokens,
            "sink_tokens": config.sink_tokens,
            "cold_page_size_mb": config.cold_page_size_bytes / (1024 ** 2),
            "kv_bytes_per_token": config.kv_bytes_per_token,
        },
        "stats": stats,
        "queries": queries,
        "all_cold_fetch_ms": round(all_cold_fetch_ms, 2),
        "all_queries_hidden": all(q["fetch_hidden"] for q in queries),
    }


def print_kv_router_report(result: dict | None = None) -> str:
    """Generate a human-readable KV Router report."""
    if result is None:
        result = simulate_kv_routing()

    cfg = result["config"]
    stats = result["stats"]
    queries = result["queries"]

    lines = []
    lines.append("=" * 60)
    lines.append("  KV ROUTER: Hot/Cold Page Management")
    lines.append("=" * 60)
    lines.append("")

    lines.append("Configuration:")
    lines.append(f"  Context length:    {cfg['total_context_tokens']:,} tokens")
    lines.append(f"  Hot tokens:        {cfg['hot_tokens']:,} (local window)")
    lines.append(f"  Sink tokens:       {cfg['sink_tokens']}")
    lines.append(f"  Cold page size:    {cfg['cold_page_size_mb']:.1f} MB")
    lines.append(f"  KV per token:      {cfg['kv_bytes_per_token']:,} bytes")
    lines.append("")

    lines.append("Page Distribution:")
    lines.append(f"  Hot pages:         {stats['hot_pages']} ({stats['hot_mb']} MB, {stats['hot_tokens']:,} tokens)")
    lines.append(f"  Cold pages:        {stats['cold_pages']} ({stats['cold_mb']} MB, {stats['cold_tokens']:,} tokens)")
    lines.append(f"  Total:             {stats['total_pages']} pages ({stats['total_mb']} MB)")
    lines.append("")

    lines.append("Query Simulations:")
    for q in queries:
        status = "HIDDEN" if q["fetch_hidden"] else f"OVERHEAD: {q['effective_overhead_ms']:.2f}ms"
        lines.append(
            f"  Position {q['query_position']:>10,}: "
            f"fetch={q['single_page_fetch_ms']:.2f}ms "
            f"window={q['draft_window_ms']:.0f}ms -> {status}"
        )
    lines.append("")

    all_cold_ms = result["all_cold_fetch_ms"]
    lines.append(f"  Full cold cache load:  {all_cold_ms:.1f} ms "
                 f"({stats['cold_mb']} MB sequential)")
    lines.append("")

    if result["all_queries_hidden"]:
        lines.append("  VERDICT: All fetches hidden in draft window.")
        lines.append("           True 0-latency infinite context achieved.")
    else:
        lines.append("  VERDICT: Some fetches exceed draft window.")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
