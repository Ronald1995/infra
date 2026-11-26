"""Lueelu ElasticBuffer expanded dispatch/combine training demo."""
from deep_ep import ElasticBuffer
from ep.deepep_ascend.demo import expanded_forward
from ep.runner import run


def forward(layer, x, ids, weights):
    return expanded_forward(layer, x, ids, weights, ElasticBuffer, use_1g_huge_pages=False)


if __name__ == "__main__":
    run(forward, "deepep_lueelu")
