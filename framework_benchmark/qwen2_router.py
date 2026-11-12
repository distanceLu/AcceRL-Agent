"""Match Qwen2-MoE's FP32 expert selection, without changing gate matmul dtype."""


def install(args=None):
    from megatron.core.transformer.moe import router
    original = router.topk_routing_with_score_function
    if getattr(original, '_qwen2_selection', False):
        return

    def routing(logits, topk, **kwargs):
        if kwargs.get('use_pre_softmax') and kwargs.get('score_function', 'softmax') == 'softmax':
            probabilities, mapping = original(logits.float(), topk, **kwargs)
            return probabilities.to(logits.dtype), mapping
        return original(logits, topk, **kwargs)

    routing._qwen2_selection = True
    router.topk_routing_with_score_function = routing
