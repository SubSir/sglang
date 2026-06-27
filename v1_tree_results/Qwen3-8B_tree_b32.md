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
- skip_baseline: `True`

Note: This sweep focuses on throughput. Correctness is not evaluated for this generic dataset script.

# Results for Dataset: `gsm8k`
## Backend: `flashinfer`

### Baseline output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 274.71 | 2,060.72 | 5,481.09 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 8.6 | 10.0 | 14.2 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2368 | 20537 | 78034 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.295 | 7.749 | 7.837 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 8.6 | 10.0 | 14.2 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2368 | 20537 | 78034 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.295 | 7.749 | 7.837 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 327 | 2716 | 10243 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

# Results for Dataset: `mt-bench`
## Backend: `flashinfer`

### Baseline output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH output tok/s
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 341.77 | 3,071.90 | 4,812.31 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 4.4 | 9.4 | 26.6 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1499 | 28823 | 127767 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2.860 | 5.557 | 5.010 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 4.4 | 9.4 | 26.6 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1499 | 28823 | 127767 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2.860 | 5.557 | 5.010 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 537 | 5789 | 29362 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

