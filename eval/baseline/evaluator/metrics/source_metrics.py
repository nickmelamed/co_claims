class SourceMetrics:

    def srs(self, domains):
        if not domains:
            return 0

        return len(set(domains)) / len(domains)

    def evs(self, external_flags):
        if not external_flags:
            return 0

        return sum(external_flags) / len(external_flags)