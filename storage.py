CHECKPOINT_SHARE_LINKS = {
    # Main result
    "CIFAR10IR100__epoch_159.pt": "10gy-6Rte8DRx-5bN-prOMui7MhQPJFuW", # Run 5jtrlqhf
    "CIFAR10IR100__epoch_199.pt": "1s92Cw4G9eA8cXKh5P8ScuGXEbmBa4SNh", # Run 5jtrlqhf
    "CIFAR10IR100-open__epoch_199.pt": "1CPl779391DObXu5iJdS6J7KzeucLjSep", # Run 76iezxbc
    "simclr-drs__epoch_199.pt": "1R2g_OcOG1XgRY6oDxXMqiC1cKoNJ-CQb", # Run wnvl5pmb

    # analysis_group_accuracy.ipynb
    "CIFAR100IR100__epoch_199.pt": "1UwEt6VJklIGsG4SVNENv9aSqUfruyvIh", # Run eqieybqp
    "CIFAR100IR100-rs__epoch_199.pt": "1bdsOBDmdE4rPUjnxq7hwmXVEYFsqk5dH", # Run vkf1unb2
    "CIFAR100IR100-open__epoch_199.pt": "1xo7ex0cIanR-FxQ8qFlKSR0fv3PdUmXk", # Run o3zqupwn
}

CHECKPOINT_URLS = {
    k: f"https://drive.google.com/uc?id={v}"
    for k, v in CHECKPOINT_SHARE_LINKS.items()
}
