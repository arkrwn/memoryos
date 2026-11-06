# ADR 0004: Quarantine low-trust facts before recall

Status: accepted

Research shows memory poisoning persists across sessions and that prompt-injection defenses do not
cover it. External content therefore never reaches recall without approval, and a lower-trust
statement cannot replace a higher-trust fact. The cost is an approval step for external sources.
