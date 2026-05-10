from tinybird_sdk import define_datasource, define_endpoint, engine, node, p, t


# --- Datasources ---

page_views = define_datasource("page_views", {
    "description": "Page view tracking data",
    "schema": {
        "timestamp": t.date_time(),
        "pathname": t.string(),
        "session_id": t.string(),
        "country": t.string().low_cardinality().nullable(),
    },
    "engine": engine.merge_tree({
        "sorting_key": ["pathname", "timestamp"],
    }),
})


# --- Endpoints ---

top_pages = define_endpoint("top_pages", {
    "description": "Get the most visited pages",
    "params": {
        "start_date": p.date_time(),
        "end_date": p.date_time(),
        "limit": p.int32().optional(10),
    },
    "nodes": [
        node({
            "name": "aggregated",
            "sql": """
                SELECT pathname, count() AS views
                FROM page_views
                WHERE timestamp >= {{DateTime(start_date)}}
                  AND timestamp <= {{DateTime(end_date)}}
                GROUP BY pathname
                ORDER BY views DESC
                LIMIT {{Int32(limit, 10)}}
            """,
        }),
    ],
    "output": {
        "pathname": t.string(),
        "views": t.uint64(),
    },
})
