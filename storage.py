CHECKPOINT_SHARE_LINKS = {
    "CIFAR10IR100__epoch_159.pt": "10gy-6Rte8DRx-5bN-prOMui7MhQPJFuW", # Run 5jtrlqhf
    "CIFAR10IR100__epoch_199.pt": "1s92Cw4G9eA8cXKh5P8ScuGXEbmBa4SNh", # Run 5jtrlqhf
    "CIFAR10IR100-open__epoch_199.pt": "1CPl779391DObXu5iJdS6J7KzeucLjSep", # Run 76iezxbc
    "simclr-drs__epoch_199.pt": "1R2g_OcOG1XgRY6oDxXMqiC1cKoNJ-CQb", # Run wnvl5pmb
}

CHECKPOINT_URLS = {
    k: f"https://drive.google.com/uc?id={v}"
    for k, v in CHECKPOINT_SHARE_LINKS.items()
}
