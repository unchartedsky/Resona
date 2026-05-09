#define NOMINMAX

#include <csignal>
#include <cstdio>
#include <cstdlib>

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"

#include "Runtime/AppRuntime.h"

namespace
{
AppRuntime *g_appRuntime = nullptr;
}

// === Signal Handler ===
void signal_handler(int signal)
{
    if (signal == SIGINT && g_appRuntime)
    {
        printf("\n[!] Caught Ctrl+C, shutting down...\n");
        g_appRuntime->RequestStop();
    }
}

int main()
{
    AppRuntime appRuntime;
    g_appRuntime = &appRuntime;
    std::signal(SIGINT, signal_handler);

    if (!appRuntime.Initialize())
    {
        g_appRuntime = nullptr;
        return EXIT_FAILURE;
    }

    const int exitCode = appRuntime.Run();
    appRuntime.Shutdown();
    g_appRuntime = nullptr;
    return exitCode;
}
