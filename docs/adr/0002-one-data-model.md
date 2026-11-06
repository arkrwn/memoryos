# ADR 0002: One data model for all capabilities

Status: accepted

A fact is also a graph edge with valid time and knowledge time, so personalization, temporal
queries and the shared graph read the same rows. Separate subsystems would need syncing and could
disagree. Agent blocks are separate because they are edited directly and are not derived from episodes.
