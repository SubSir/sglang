# DFLASH Sweep: gsm8k, mt-bench

## Settings
- data_names: `gsm8k, mt-bench`
- target_model: `Qwen/Qwen3-8B`
- draft_model: `z-lab/Qwen3-8B-DFlash-b16`
- prompt_style: `chat`
- max_new_tokens: `1024`
- attention_backends: `flashinfer`
- tp_sizes: `1`
- concurrencies: `1, 8, 32`
- samples_per_concurrency: `base=8`
- device_sm: `100`
- is_blackwell: `True`
- skip_baseline: `False`

Note: This sweep focuses on throughput. Correctness is not evaluated for this generic dataset script.

# Results for Dataset: `gsm8k`
## Backend: `flashinfer`

### Baseline output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 226.67 | 1,413.30 | 4,900.94 |

### DFLASH output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 319.10 | 3,649.06 | 8,935.41 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1.408 | 2.582 | 1.823 |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.2 | 5.5 | 8.7 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2307 | 19990 | 78090 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 5.867 | 6.388 | 6.407 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1.408 | 2.582 | 1.823 |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.2 | 5.5 | 8.7 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2307 | 19990 | 78090 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 5.867 | 6.388 | 6.407 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 399 | 3237 | 12659 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

# Results for Dataset: `mt-bench`
## Backend: `flashinfer`

### Baseline output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 235.18 | 1,473.95 | 5,281.16 |

### DFLASH output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 293.40 | 2,862.09 | 5,594.23 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1.248 | 1.942 | 1.059 |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 5.3 | 9.9 | 22.3 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1552 | 28370 | 124744 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2.196 | 4.525 | 3.966 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1.248 | 1.942 | 1.059 |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 5.3 | 9.9 | 22.3 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1552 | 28370 | 124744 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2.196 | 4.525 | 3.966 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 740 | 7394 | 37655 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

