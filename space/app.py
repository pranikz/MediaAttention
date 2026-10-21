from mediaattention.ui import build

build().queue(default_concurrency_limit=1).launch()
