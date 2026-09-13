def build_adjacency_index(graph):
    """Builds a bidirectional adjacency index from graph edges."""
    adj = {}
    for edge in graph["edges"]:
        s, t, etype = edge["source"], edge["target"], edge["type"]
        adj.setdefault(s, []).append((t, etype))
        adj.setdefault(t, []).append((s, etype + "_REV"))
    return adj

def get_last_qa(analyst, graph, train_quarters, adj, quarter_sort_map):
    """Returns the most recent question node and its answer node for `analyst`."""
    qs = [n for n in graph["nodes"]
          if n["type"] == "Question"
          and n["properties"]["analyst"] == analyst
          and n["properties"]["quarter"] in train_quarters]
    if not qs:
        return None, None
    qs_sorted = sorted(qs, key=lambda x: quarter_sort_map.get(x["properties"]["quarter"], 0), reverse=True)
    last_q = qs_sorted[0]
    q_id   = last_q["id"]
    for nbr, etype in adj.get(q_id, []):
        if etype in ("RESPONDED_TO_REV", "RESPONDED_TO"): # check both directions for safety
            ans_nodes = [n for n in graph["nodes"] if n["id"] == nbr]
            if ans_nodes:
                return last_q, ans_nodes[0]
    return last_q, None

def get_cross_follow_context(analyst, pref, q3_questions):
    """Returns up to 2 other-analyst Q3FY26 questions on this analyst's top topics."""
    fav_topics = [t for t, _ in sorted(pref.items(), key=lambda x: x[1], reverse=True)[:3]]
    ctx = []
    for t in fav_topics:
        others = [q for q in q3_questions
                  if t in q["properties"]["topics"] and q["properties"]["analyst"] != analyst]
        for q in others[:2]:
            ctx.append({"analyst": q["properties"]["analyst"], "topic": t,
                        "question": q["properties"]["text"]})
    return ctx
