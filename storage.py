# Links to Google Drives removed for Anonymization
CHECKPOINT_SHARE_LINKS = {
    # Main result
    "CIFAR10IR100__epoch_159.pt": "ANONYMIZED", # Run 5jtrlqhf
    "CIFAR10IR100__epoch_199.pt": "ANONYMIZED", # Run 5jtrlqhf
    "CIFAR10IR100-open__epoch_199.pt": "ANONYMIZED", # Run 76iezxbc
    "simclr-drs__epoch_199.pt": "ANONYMIZED", # Run wnvl5pmb

    # analysis_group_accuracy.ipynb
    "CIFAR100IR100__epoch_199.pt": "ANONYMIZED", # Run eqieybqp
    "CIFAR100IR100-rs__epoch_199.pt": "ANONYMIZED", # Run vkf1unb2
    "CIFAR100IR100-open__epoch_199.pt": "ANONYMIZED", # Run o3zqupwn
}

CHECKPOINT_URLS = {
    k: f"https://drive.google.com/uc?id={v}"
    for k, v in CHECKPOINT_SHARE_LINKS.items()
}
