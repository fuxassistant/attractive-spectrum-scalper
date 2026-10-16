"""Materialize GCQ6 separately during the late-May liquidity transition."""
import extend_may_cluster as run

if __name__ == '__main__':
    run.RAW = run.ROOT/'data/derived/raw_may_roll_v1'
    run.MARKET = run.ROOT/'data/derived/market_may_roll_v1'
    run.FEATURES = run.ROOT/'data/derived/features_may_roll_v1'
    run.REPORT = run.ROOT/'reports/may_roll_v1'
    run.ROUTE = [1, 42011464, 22]
    run.START = '2026-05-24T00:00Z'
    run.FIRST_CAPTURE = 24
    run.BUILD_WORKERS, run.FEATURE_WORKERS = 2, 1
    run.PUBLISH_CLUSTER = False
    run.main()
