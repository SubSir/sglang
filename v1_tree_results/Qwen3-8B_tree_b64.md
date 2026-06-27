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
| 1 | 343.72 | 1,966.89 | 4,280.98 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 6.9 | 10.1 | 18.1 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2383 | 19903 | 77540 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.742 | 8.123 | 8.218 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 6.9 | 10.1 | 18.1 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2383 | 19903 | 77540 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 7.742 | 8.123 | 8.218 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 310 | 2496 | 9672 |

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
| 1 | 332.75 | 2,477.85 | 3,250.53 |

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 4.4 | 11.5 | 38.8 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1476 | 28550 | 125993 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2.966 | 5.763 | 5.250 |

---

### Speedup (DFLASH / baseline)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | N/A | N/A | N/A |

### DFLASH latency (seconds)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 4.4 | 11.5 | 38.8 |

### DFLASH output tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 1476 | 28550 | 125993 |

### DFLASH acceptance length (mean per-request spec_accept_length)
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 2.966 | 5.763 | 5.250 |

### DFLASH total forward count
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 514 | 5517 | 27713 |

### DFLASH total verified tokens
| tp\conc | 1 | 8 | 32 |
| --- | --- | --- | --- |
| 1 | 0 | 0 | 0 |

