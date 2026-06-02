#include "StatusReporter.h"

#include <algorithm>
#include <cstdio>

void StatusReporter::PrintStatusLine(const RuntimeStatusSnapshot &snapshot)
{
    const float outputFillPercent =
        snapshot.outputRingCapacityFrames > 0
            ? (snapshot.outputBufferFrames * 100.0f) / static_cast<float>(snapshot.outputRingCapacityFrames)
            : 0.0f;

    const float ratioDeviation =
        snapshot.baseRatio > 0.0f ? ((snapshot.currentRatio / snapshot.baseRatio) - 1.0f) * 100.0f : 0.0f;

    const float targetFillPercent = std::clamp(snapshot.targetFillRatio, 0.0f, 1.0f) * 100.0f;
    const uint32_t minOutputFrames = snapshot.minObservedOutputFrames;

    printf("\r\033[K");
    printf("Out:%u(%.0f%%%+.0f) Min:%u GPU:%u/%u Ratio:%.6f(%+.3f%%) Target:%.0f%% ZF:%llu %llus",
            snapshot.outputBufferFrames, outputFillPercent, snapshot.outputFillRate, minOutputFrames,
            snapshot.busySlots, snapshot.totalSlots, snapshot.currentRatio, ratioDeviation, targetFillPercent,
            static_cast<unsigned long long>(snapshot.zeroFillEvents),
           static_cast<unsigned long long>(snapshot.totalElapsedSeconds));
    fflush(stdout);
}

void StatusReporter::PrintSessionStatistics(const SessionStatistics &statistics)
{
    if (statistics.totalElapsedSeconds <= 0)
    {
        return;
    }

    const double elapsedSeconds = static_cast<double>(statistics.totalElapsedSeconds);

    printf("[+] Session statistics:\n");
    printf("    Runtime: %lld seconds\n", statistics.totalElapsedSeconds);
    printf("    Input captured: %llu frames (%.1f fps avg)\n", statistics.totalCapturedFrames,
           statistics.totalCapturedFrames / elapsedSeconds);
    printf("    Input processed: %llu frames (%.1f fps avg)\n", statistics.totalProcessedFrames,
           statistics.totalProcessedFrames / elapsedSeconds);
    printf("    Output played: %llu frames (%.1f fps avg)\n", statistics.totalPlayedFrames,
           statistics.totalPlayedFrames / elapsedSeconds);
    printf("    Lowest output buffer: %u frames\n", statistics.minObservedOutputFrames);
    printf("    Zero-fill events: %llu\n", statistics.zeroFillEvents);
    printf("    Zero-fill samples: %llu\n", statistics.zeroFillSamples);
}
