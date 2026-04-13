from __future__ import annotations

from src.app_config import ensure_default_config, load_config
from src.pipeline_runner import PipelineRunner


def main() -> int:
    ensure_default_config()
    config = load_config()
    runner = PipelineRunner(config, logger=print)
    result = runner.run()
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())

