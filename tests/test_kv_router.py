"""Tests for neuralbyte.spec_decode.kv_router — hot/cold KV page management."""
import pytest

from neuralbyte.spec_decode.hardware import HardwareProfile
from neuralbyte.spec_decode.kv_router import (
    KVPage,
    KVRouter,
    KVRouterConfig,
    print_kv_router_report,
    simulate_kv_routing,
)


# ── KVRouterConfig ────────────────────────────────────────────

class TestKVRouterConfig:
    def test_defaults(self):
        cfg = KVRouterConfig()
        assert cfg.total_context_tokens == 1_000_000
        assert cfg.hot_tokens == 2048
        assert cfg.sink_tokens == 4

    def test_tokens_per_cold_page(self):
        cfg = KVRouterConfig()
        # 2 MB / 20480 bytes per token = 102 tokens
        assert cfg.tokens_per_cold_page == 2 * 1024 * 1024 // 20480

    def test_cold_tokens(self):
        cfg = KVRouterConfig()
        expected = cfg.total_context_tokens - cfg.hot_tokens - cfg.sink_tokens
        assert cfg.cold_tokens == expected

    def test_total_kv_bytes(self):
        cfg = KVRouterConfig()
        assert cfg.total_kv_bytes == cfg.total_context_tokens * cfg.kv_bytes_per_token

    def test_hot_kv_bytes(self):
        cfg = KVRouterConfig()
        hot = cfg.hot_tokens + cfg.sink_tokens
        assert cfg.hot_kv_bytes == hot * cfg.kv_bytes_per_token

    def test_cold_kv_bytes(self):
        cfg = KVRouterConfig()
        assert cfg.cold_kv_bytes == cfg.cold_tokens * cfg.kv_bytes_per_token

    def test_n_cold_pages_positive(self):
        cfg = KVRouterConfig()
        assert cfg.n_cold_pages > 0

    def test_zero_kv_bytes_per_token(self):
        cfg = KVRouterConfig(kv_bytes_per_token=0)
        assert cfg.tokens_per_cold_page == 0
        assert cfg.n_cold_pages == 0


# ── KVPage ────────────────────────────────────────────────────

class TestKVPage:
    def test_n_tokens(self):
        page = KVPage(page_id=0, token_start=100, token_end=200, location="ram")
        assert page.n_tokens == 100

    def test_token_range(self):
        page = KVPage(page_id=0, token_start=10, token_end=50, location="ssd")
        assert page.token_range == (10, 50)


# ── KVRouter.from_context ────────────────────────────────────

class TestKVRouterFromContext:
    def test_builds_pages(self):
        router = KVRouter.from_context()
        assert len(router.pages) > 0
        router.shutdown()

    def test_sink_page_in_ram(self):
        router = KVRouter.from_context()
        sink = router.pages[0]
        assert sink.location == "ram"
        assert sink.token_start == 0
        assert sink.token_end == 4
        router.shutdown()

    def test_local_window_in_ram(self):
        router = KVRouter.from_context()
        last_page = router.pages[-1]
        assert last_page.location == "ram"
        assert last_page.token_end == 1_000_000
        router.shutdown()

    def test_cold_pages_on_ssd(self):
        router = KVRouter.from_context()
        cold = [p for p in router.pages if p.location == "ssd"]
        assert len(cold) > 0
        router.shutdown()

    def test_hot_cold_split(self):
        router = KVRouter.from_context()
        assert len(router.hot_pages) >= 2  # sink + local window
        assert len(router.cold_pages) > 0
        router.shutdown()

    def test_small_context_no_cold(self):
        cfg = KVRouterConfig(total_context_tokens=100, hot_tokens=90, sink_tokens=10)
        router = KVRouter.from_context(cfg)
        assert len(router.cold_pages) == 0
        router.shutdown()

    def test_custom_hardware(self):
        hw = HardwareProfile(ssd_sequential_gb_s=3.5)
        router = KVRouter.from_context(hw=hw)
        assert len(router.pages) > 0
        router.shutdown()


# ── KVRouter.route ────────────────────────────────────────────

class TestKVRouterRoute:
    def test_route_returns_hot_and_cold(self):
        router = KVRouter.from_context()
        hot, cold = router.route(500_000)
        assert len(hot) > 0
        assert len(cold) > 0
        router.shutdown()

    def test_all_pages_accounted(self):
        router = KVRouter.from_context()
        hot, cold = router.route(500_000)
        assert len(hot) + len(cold) == len(router.pages)
        router.shutdown()


# ── KVRouter.fetch ────────────────────────────────────────────

class TestKVRouterFetch:
    def test_sync_fetch_promotes_to_ram(self):
        cfg = KVRouterConfig(total_context_tokens=500, hot_tokens=100, sink_tokens=4)
        router = KVRouter.from_context(cfg)
        cold = router.cold_pages[:1]
        if cold:
            fetched = router.fetch_cold_pages_sync(cold)
            assert len(fetched) == 1
            assert fetched[0].location == "ram"
            assert fetched[0].data is not None
        router.shutdown()

    def test_async_fetch_returns_future(self):
        cfg = KVRouterConfig(total_context_tokens=500, hot_tokens=100, sink_tokens=4)
        router = KVRouter.from_context(cfg)
        cold = router.cold_pages[:1]
        if cold:
            future = router.fetch_cold_pages_async(cold)
            result = future.result(timeout=5.0)
            assert len(result) == 1
            assert result[0].location == "ram"
        router.shutdown()


# ── KVRouter.evict / promote ─────────────────────────────────

class TestKVRouterEvictPromote:
    def test_evict_cold_pages(self):
        cfg = KVRouterConfig(total_context_tokens=500, hot_tokens=100, sink_tokens=4)
        router = KVRouter.from_context(cfg)
        cold = router.cold_pages[:1]
        if cold:
            router.fetch_cold_pages_sync(cold)
            assert cold[0].location == "ram"
            evicted = router.evict_to_ssd(cold)
            assert evicted == 1
            assert cold[0].location == "ssd"
        router.shutdown()

    def test_cannot_evict_sink(self):
        router = KVRouter.from_context()
        sink = [p for p in router.pages if p.token_start == 0]
        evicted = router.evict_to_ssd(sink)
        assert evicted == 0
        router.shutdown()

    def test_cannot_evict_local_window(self):
        router = KVRouter.from_context()
        local = [router.pages[-1]]
        evicted = router.evict_to_ssd(local)
        assert evicted == 0
        router.shutdown()

    def test_promote_pages(self):
        cfg = KVRouterConfig(total_context_tokens=500, hot_tokens=100, sink_tokens=4)
        router = KVRouter.from_context(cfg)
        cold = router.cold_pages[:1]
        if cold:
            promoted = router.promote_to_ram(cold)
            assert promoted == 1
            assert cold[0].location == "ram"
        router.shutdown()


# ── KVRouter.stats ────────────────────────────────────────────

class TestKVRouterStats:
    def test_stats_keys(self):
        router = KVRouter.from_context()
        stats = router.stats()
        for key in ("total_pages", "hot_pages", "cold_pages", "hot_mb", "cold_mb", "total_mb"):
            assert key in stats
        router.shutdown()

    def test_hot_cold_bytes_sum(self):
        router = KVRouter.from_context()
        stats = router.stats()
        assert stats["hot_pages"] + stats["cold_pages"] == stats["total_pages"]
        router.shutdown()

    def test_hot_tokens_equals_sink_plus_window(self):
        router = KVRouter.from_context()
        stats = router.stats()
        cfg = router.config
        assert stats["hot_tokens"] == cfg.sink_tokens + cfg.hot_tokens
        router.shutdown()


# ── KVRouter.simulate_query ──────────────────────────────────

class TestKVRouterSimulateQuery:
    def test_fetch_hidden_in_draft_window(self):
        router = KVRouter.from_context()
        result = router.simulate_query()
        assert result["fetch_hidden"] is True
        router.shutdown()

    def test_single_page_fetch_under_1ms(self):
        router = KVRouter.from_context()
        result = router.simulate_query()
        assert result["single_page_fetch_ms"] < 1.0
        router.shutdown()

    def test_zero_overhead_when_hidden(self):
        router = KVRouter.from_context()
        result = router.simulate_query()
        assert result["effective_overhead_ms"] == 0.0
        router.shutdown()

    def test_custom_query_position(self):
        router = KVRouter.from_context()
        result = router.simulate_query(query_position=500_000)
        assert result["query_position"] == 500_000
        router.shutdown()


# ── simulate_kv_routing ──────────────────────────────────────

class TestSimulateKvRouting:
    def test_returns_expected_keys(self):
        result = simulate_kv_routing()
        for key in ("config", "stats", "queries", "all_cold_fetch_ms", "all_queries_hidden"):
            assert key in result

    def test_all_queries_hidden(self):
        result = simulate_kv_routing()
        assert result["all_queries_hidden"] is True

    def test_four_query_positions(self):
        result = simulate_kv_routing()
        assert len(result["queries"]) == 4


# ── print_kv_router_report ───────────────────────────────────

class TestPrintKvRouterReport:
    def test_report_contains_sections(self):
        report = print_kv_router_report()
        assert "KV ROUTER" in report
        assert "Configuration" in report
        assert "Page Distribution" in report
        assert "Query Simulations" in report

    def test_report_shows_verdict(self):
        report = print_kv_router_report()
        assert "VERDICT" in report

    def test_report_shows_infinite_context(self):
        report = print_kv_router_report()
        assert "infinite context" in report.lower()
