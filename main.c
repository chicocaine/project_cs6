#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <omp.h>
#include <sys/stat.h>
#include <malloc.h>
#include "standard.h"
#include "strassen.h"
#include "standard_block.h"

static int* allocate_matrix(int n) {
    int *m = malloc((size_t)n * n * sizeof *m);
    if (!m) { perror("malloc"); exit(EXIT_FAILURE); }
    return m;
}

static void fill_matrix(int* m, int n) {
    #pragma omp parallel
    {
        unsigned int seed = time(NULL) ^ omp_get_thread_num();
        #pragma omp for
        for (int i = 0; i < n * n; i++) {
            m[i] = rand_r(&seed) % 10;
        }
    }
}

static long getCurrentRSS_kB(void) {
    FILE *f = fopen("/proc/self/status","r");
    if (!f) return 0;
    char line[256];
    long rss = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "VmRSS: %ld kB", &rss) == 1) break;
    }
    fclose(f);
    return rss;
}

static void check_folder(const char* folder) {
    struct stat st;
    if (stat(folder, &st) == -1) {
        if (mkdir(folder, 0755) == -1) {
            perror("mkdir");
            exit(EXIT_FAILURE);
        }
    } else if (!S_ISDIR(st.st_mode)) {
        fprintf(stderr, "%s is not a directory\n", folder);
        exit(EXIT_FAILURE);
    }
}

static char* unique_filename(const char* base, const char* ext) {
    char *name = malloc(strlen(base) + strlen(ext) + 32);
    if (!name) { perror("malloc"); exit(EXIT_FAILURE); }
    sprintf(name, "%s.%s", base, ext);
    struct stat st;
    int count = 1;
    while (stat(name, &st) == 0) {
        free(name);
        name = malloc(strlen(base) + strlen(ext) + 32);
        sprintf(name, "%s_%d.%s", base, count++, ext);
    }
    return name;
}

int main(int argc, char *argv[]) {
    int PASSES = 0, POWER = 0;
    int THRESHOLD = 0, BLOCKSIZE = 0;
    int STANDARD = 0;
    int STRASSEN = 0;
    int THREADCOUNT = 1;
    int TIME_DISABLE = 0, MEMORY_DISABLE = 0, CHECK_CORRECTNESS = 0;
    int VERBOSE = 0, HELP = 0;
    char *OUT_BASE = NULL;
    char *OUT_EXT = NULL;
    char *OUT_FOLDER = "results";

    static struct option long_opts[] = {
        {"passes", required_argument, 0, 'p'},
        {"power", required_argument, 0, 'w'},
        {"threshold", required_argument, 0, 't'},
        {"blocksize", required_argument, 0, 'b'},
        {"standard", no_argument, 0, 'n'},
        {"strassen", no_argument, 0, 's'},
        {"threadcount", required_argument, 0, 'T'},
        {"time-disable", no_argument, 0, '1'},
        {"memory-disable", no_argument, 0, '2'},
        {"check-correctness", no_argument, 0, '3'},
        {"help", no_argument, 0, 'h'},
        {"verbose", no_argument, 0, 'v'},
        {"output", required_argument, 0, 'o'},
        {0, 0, 0, 0}
    };

    int opt, idx;
    while ((opt = getopt_long(argc, argv, "p:w:t:b:nsT:123hvo:", long_opts, &idx)) != -1) {
        switch (opt) {
            case 'p': PASSES = atoi(optarg); break;
            case 'w': POWER = atoi(optarg); break;
            case 't': THRESHOLD = atoi(optarg); break;
            case 'b': BLOCKSIZE = atoi(optarg); break;
            case 'n': STANDARD = 1; break;
            case 's': STRASSEN = 1; break;
            case 'T': THREADCOUNT = atoi(optarg); break;
            case '1': TIME_DISABLE = 1; break;
            case '2': MEMORY_DISABLE = 1; break;
            case '3': CHECK_CORRECTNESS = 1; break;
            case 'v': VERBOSE = 1; break;
            case 'h': HELP = 1; break;
            case 'o': {
                char *arg = strdup(optarg);
                char *dot = strrchr(arg, '.');
                if (dot) {
                    *dot = '\0';
                    OUT_BASE = strdup(arg);
                    OUT_EXT = strdup(dot + 1);
                } else {
                    OUT_BASE = arg;
                }
                break;
            }
            default:
                fprintf(stderr, "Use -h for help\n");
                exit(EXIT_FAILURE);
        }
    }

    if (HELP || argc == 1) {
        printf("Usage: %s\n"
               "  -p passes -w power\n"
               " [options]\n"
               "  -t threshold   use Strassen threshold t[input]>0\n"
               "  -b blocksize   use blocked algorithm b[input]>0\n"
               "  -n standard    runs standard algorithm count\n"
               "  -s strassen    runs Strassen count\n"
               "  -T threads     set thread count [default: 1]\n"
               "  -1             disable timing\n"
               "  -2             disable memory logging\n"
               "  -3             check correctness\n"
               "  -o file[.json|.csv|.txt]  output base/name and format\n",
               argv[0]);
        exit(EXIT_SUCCESS);
    }

    if (PASSES < 1 || POWER < 1) {
        fprintf(stderr, "Error: passes and power are required\n");
        exit(EXIT_FAILURE);
    }

    if (!OUT_BASE) {
        OUT_BASE = strdup("results");
        OUT_EXT = strdup("json");
    }

    if (OUT_EXT) {
        if (strcmp(OUT_EXT, "json") != 0 && strcmp(OUT_EXT, "csv") != 0 && strcmp(OUT_EXT, "txt") != 0) {
            fprintf(stderr, "Error: output format must be json, csv or txt\n");
            exit(EXIT_FAILURE);
        }
    } else {
        OUT_EXT = strdup("json");
    }

    if (OUT_BASE) {
        check_folder(OUT_FOLDER);
        char *full_path = malloc(strlen(OUT_FOLDER) + strlen(OUT_BASE) + 2);
        sprintf(full_path, "%s/%s", OUT_FOLDER, OUT_BASE);
        free(OUT_BASE);
        OUT_BASE = full_path;
    }

    char *filename = unique_filename(OUT_BASE, OUT_EXT);
    FILE *out = fopen(filename, "w");
    if (!out) { perror("fopen"); exit(EXIT_FAILURE); }

    int first_record = 1;
    if (strcmp(OUT_EXT, "json") == 0) {
        fprintf(out, "{\n  \"results\": [\n");
    } else if (strcmp(OUT_EXT, "csv") == 0) {
        fprintf(out, "algorithm,n,time_s,rss_kB,equivalent\n");
    }

    printf("Starting matrix multiplication benchmark...\n");
    if (VERBOSE && !HELP) {
        printf("=========================================\n");
        printf("PASSES: %d\n", PASSES);
        printf("POWER: %d\n", POWER);
        printf("THRESHOLD: %d\n", THRESHOLD);
        printf("BLOCKSIZE: %d\n", BLOCKSIZE);
        printf("STANDARD: %d\n", STANDARD);
        printf("STRASSEN: %d\n", STRASSEN);
        printf("THREADCOUNT: %d\n", THREADCOUNT);
        printf("TIME_DISABLE: %d\n", TIME_DISABLE);
        printf("MEMORY_DISABLE: %d\n", MEMORY_DISABLE);
        printf("CHECK_CORRECTNESS: %d\n", CHECK_CORRECTNESS);
        printf("==========================================\n");
    }

    srand((unsigned)time(NULL));
    for (int pass = 0; pass <= PASSES; pass++) {
        printf("Pass: %d/%d\n", pass, PASSES);

        for (int j = 0; j <= POWER; j++) {
            int n = 1 << j;
            printf("\x1b[2K\r");
            printf("Testing matrix size: %d (2^%d)...\n", n, j);

            int *A = allocate_matrix(n), *B = allocate_matrix(n);
            int *Cstd = allocate_matrix(n), *Cblk = allocate_matrix(n), *Cstr = allocate_matrix(n);
            fill_matrix(A, n); fill_matrix(B, n);

            double t_std = 0, t_blk = 0, t_str = 0;
            long rss_std = 0, rss_blk = 0, rss_str = 0;
            struct timespec t0, t1;

            if (STANDARD) {
                standard_multiply(A, B, Cstd, n, THREADCOUNT);
                if (!TIME_DISABLE) clock_gettime(CLOCK_MONOTONIC, &t0), standard_multiply(A, B, Cstd, n, THREADCOUNT), clock_gettime(CLOCK_MONOTONIC, &t1), t_std = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
                if (!MEMORY_DISABLE) rss_std = getCurrentRSS_kB();
            }
            if (BLOCKSIZE) {
                memset(Cblk, 0, n*n*sizeof(int));
                blocked_multiply(A, B, Cblk, n, BLOCKSIZE, THREADCOUNT);
                if (!TIME_DISABLE) clock_gettime(CLOCK_MONOTONIC, &t0), blocked_multiply(A, B, Cblk, n, BLOCKSIZE, THREADCOUNT), clock_gettime(CLOCK_MONOTONIC, &t1), t_blk = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
                if (!MEMORY_DISABLE) rss_blk = getCurrentRSS_kB();
            }
            if (STRASSEN) {
                #pragma omp parallel num_threads(THREADCOUNT)
                #pragma omp single
                {
                    strassen_rec(A, B, Cstr, n, THRESHOLD, THREADCOUNT);
                    if (!TIME_DISABLE) clock_gettime(CLOCK_MONOTONIC, &t0), strassen_rec(A, B, Cstr, n, THRESHOLD, THREADCOUNT), clock_gettime(CLOCK_MONOTONIC, &t1), t_str = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
                    if (!MEMORY_DISABLE) rss_str = getCurrentRSS_kB();
                }
            }

            int eq = 1;
            if (CHECK_CORRECTNESS) {
                eq = !memcmp(Cstd, (STRASSEN ? Cstr : Cblk), n*n*sizeof(int));
            }

            if (strcmp(OUT_EXT, "json") == 0) {
                if (!first_record) fprintf(out, ",\n");
                fprintf(out, "    { \"pass\": %d, \"n\": %d,", pass, n);
                if (STANDARD) {
                    fprintf(out, " \"standard\": { \"time_s\": %.9f, \"rss_kB\": %ld },", t_std, rss_std);
                }
                if (BLOCKSIZE) {
                    fprintf(out, " \"blocked\": { \"time_s\": %.9f, \"rss_kB\": %ld },", t_blk, rss_blk);
                }
                if (STRASSEN) {
                    fprintf(out, " \"strassen\": { \"time_s\": %.9f, \"rss_kB\": %ld },", t_str, rss_str);
                }
                fprintf(out, " \"equivalent\": %s }", eq ? "true" : "false");
                first_record = 0;
            } else if (strcmp(OUT_EXT, "csv")==0) {
                if (STANDARD) {
                    fprintf(out, "standard,%d,%.9f,%ld,%d\n", n, t_std, rss_std, eq);
                }
                if (BLOCKSIZE) {
                    fprintf(out, "blocked,%d,%.9f,%ld,%d\n", n, t_blk, rss_blk, eq);
                }
                if (STRASSEN) {
                    fprintf(out, "strassen,%d,%.9f,%ld,%d\n", n, t_str, rss_str, eq);
                }
            } else if (strcmp(OUT_EXT, "txt")==0) {
                if (STANDARD) {
                    fprintf(out, "standard: n=%d time=%.9fs rss=%ldkB eq=%d\n", n, t_std, rss_std, eq);
                }
                if (BLOCKSIZE) {
                    fprintf(out, "blocked: n=%d time=%.9fs rss=%ldkB eq=%d\n", n, t_blk, rss_blk, eq);
                }
                if (STRASSEN) {
                    fprintf(out, "strassen: n=%d time=%.9fs rss=%ldkB eq=%d\n", n, t_str, rss_str, eq);
                }
            }

            free(A); free(B);
            free(Cstd); free(Cblk); free(Cstr);
            malloc_trim(0);
        }
    }

    if (strcmp(OUT_EXT, "json")==0) fprintf(out, "\n  ]\n}\n");
    fclose(out);
    if (VERBOSE) printf("Results written to %s\n", filename);

    if (strcmp(OUT_EXT, "json")==0) {
        size_t buf_size = strlen(filename) + sizeof("python3 visualizer.py --file ") + 1;
        char *command = malloc(buf_size);
        if (!command) {
            perror("malloc");
            return EXIT_FAILURE;
        }
    
        int needed = snprintf(command, buf_size, "python3 visualizer.py --file %s", filename);
        if (needed < 0 || (size_t)needed >= buf_size) {
            fprintf(stderr, "Formatting error or buffer too small\n");
            free(command);
            return EXIT_FAILURE;
        }
    
        printf("Running visualizer.py...\n");
        if (VERBOSE) printf("Command: %s\n", command);
        int ret = system(command);
        free(command);
        if (ret == -1) {
            perror("system");
            return EXIT_FAILURE;
        }
        if (WEXITSTATUS(ret) != 0) {
            fprintf(stderr, "visualizer.py exited with %d\n",
                    WEXITSTATUS(ret));
            return EXIT_FAILURE;
        }
    }

    free(filename); free(OUT_BASE); free(OUT_EXT);
    
    return 0;
}