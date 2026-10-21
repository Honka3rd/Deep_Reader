"""Content-free accounting of the actual serialized planner request."""

import json
import logging

LOGGER = logging.getLogger(__name__)


def prompt_metrics(prompt, provider=None, token_counter=None):
    instructions, schema_context = prompt.split("SCHEMA:\n", 1)
    schema, context_text = schema_context.split("\nCONTEXT:\n", 1)
    context, context_end = json.JSONDecoder().raw_decode(context_text)
    if context["source"].get("representation") == "readable_full_source_v1":
        from document_structure.semantic_structure_plan import read_semantic_prompt_context
        context = read_semantic_prompt_context(prompt)
        source_end = context_end + len("\nSOURCE TEXT:\n") + context["source"]["readable_text_chars"]
        separator = "\nFINAL INSTRUCTIONS:\n"
        final_instructions = context_text[source_end + len(separator):]
    else:
        context_text, separator, final_instructions = context_text.partition("\nFINAL INSTRUCTIONS:\n")
    source = context["source"]
    plain = source.get("representation") == "readable_full_source_v1"
    segments = source.get("segments", [])
    located = source.get("representation") == "located_source_passages_v2"
    compact = plain or located or source.get("representation") == "readable_reference_map_v1"
    dump = lambda value: json.dumps(value, ensure_ascii=False, **({"separators": (",", ":")} if compact else {}))
    # String costs include JSON escaping but exclude the surrounding quotes.
    string_cost = lambda value: len(json.dumps(value, ensure_ascii=False)) - 2
    parts = {
        "instructions": len(instructions) + len(separator) + len(final_instructions),
        "schema": len(schema),
        "source_text": len(source["text"]) if plain else sum(string_cost(p["text"]) for s in segments for p in s["passages"]) if located
                       else sum(string_cost(s["text"]) for s in segments),
        "analysis_text": sum(string_cost(s.get("analysis_text", "")) for s in segments),
        "references": sum(len(dump(p["boundary_id"])) for s in segments for p in s["passages"]
                          if p["boundary_id"] is not None) if located
                      else sum(len(dump(s["boundary_references"])) for s in segments),
        "common_structure": len(dump(context["common_structure"])) if "common_structure" in context else 0,
    }
    parts["serialization_metadata"] = len(prompt) - sum(parts.values())
    visible = source["source_length"] if plain else sum(s["source_end"] - s["source_start"] for s in segments)
    metrics = {
        "metrics_version": "plan_prompt_metrics_v2",
        "prompt_version": context["prompt_version"],
        "prompt_chars": len(prompt), "component_chars": parts,
        "source_chars": source["source_length"], "visible_source_chars": visible,
        "coverage_ratio": visible / max(1, source["source_length"]),
        "omitted_range_count": len(source["omitted_ranges"]),
        "segment_count": 1 if plain else len(segments),
        "reference_count": sum(p["boundary_id"] is not None for s in segments for p in s["passages"])
                           if located else sum(len(s["boundary_references"]) for s in segments),
        "reference_preview_chars": 0 if compact else sum(string_cost(r["preview"]) for s in segments
                                                        for r in s["boundary_references"]),
        "input_token_count": None,
        "input_token_estimate": len(prompt.encode("utf-8")),
        "token_measurement": "utf8_byte_upper_estimate_not_provider_usage",
        "context_window_tokens": None, "max_input_tokens": None, "max_output_tokens": None,
        "configured_output_tokens": None,
    }
    if token_counter is not None:
        try:
            count = token_counter(prompt)
            if type(count) is int and count >= 0:
                metrics.update(input_token_count=count, input_token_estimate=None,
                               token_measurement="configured_tokenizer_local_count_not_provider_usage")
        except Exception:
            pass  # Metrics must not expose tokenizer errors containing source text.
    if provider is not None:
        try:
            caps = provider.get_model_capabilities()
            for key in ("context_window_tokens", "max_input_tokens", "max_output_tokens"):
                value = getattr(caps, key, None)
                if type(value) is int and value > 0:
                    metrics[key] = value
        except Exception:
            pass
        output = getattr(provider, "effective_output_budget", None)
        if type(output) is int and output > 0:
            metrics["configured_output_tokens"] = output
    return metrics


def log_prompt_metrics(prompt, provider):
    try:
        LOGGER.info("enhancement_prompt_metrics %s", json.dumps(prompt_metrics(prompt, provider), sort_keys=True))
    except Exception:
        LOGGER.warning("enhancement_prompt_metrics_unavailable")
