# DFLASH Sweep: gsm8k, mt-bench

## Settings
- data_names: `gsm8k, mt-bench`
- target_model: `Qwen/Qwen3.6-27B`
- draft_model: `z-lab/Qwen3.6-27B-DFlash`
- prompt_style: `chat`
- max_new_tokens: `1024`
- attention_backends: `triton`
- tp_sizes: `1`
- concurrencies: `1, 8, 32`
- samples_per_concurrency: `base=8`
- device_sm: `100`
- is_blackwell: `True`
- skip_baseline: `True`

Note: This sweep focuses on throughput. Correctness is not evaluated for this generic dataset script.

# Results for Dataset: `gsm8k`
## Backend: `triton`

### Baseline output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 376.28 | 1,124.17 | 1,868.40 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.5 | 22.9 | 54.4 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2836 | 25727 | 101672 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.791 | 7.916 | 8.144 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.5 | 22.9 | 54.4 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2836 | 25727 | 101672 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.791 | 7.916 | 8.144 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 370 | 3415 | 13027 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

# Results for Dataset: `mt-bench`
## Backend: `triton`

### Baseline output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 171.86 | 1,218.71 | 1,289.08 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 11.0 | 26.4 | 113.0 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1898 | 32227 | 145697 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2.935 | 5.555 | 5.070 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 11.0 | 26.4 | 113.0 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1898 | 32227 | 145697 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2.935 | 5.555 | 5.070 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 624 | 6310 | 32366 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

