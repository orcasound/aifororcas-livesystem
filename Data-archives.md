# Data archive format

(if you have admin access, look at [this S3 location](https://s3.console.aws.amazon.com/s3/buckets/acoustic-sandbox/labeled-data/detection/))

Each file is tar compressed archive with the directory structure and details below. 
[open on Linux](https://www.howtogeek.com/248780/how-to-compress-and-extract-files-using-the-tar-command-on-linux/), [use WSL on Windows](https://docs.microsoft.com/en-us/windows/wsl/about), [other Windows option](https://www.7-zip.org/)

Standard dataset format understood by the `AudioFileDataset` class consists of:
* A train/dev TSV file containing at minimum, columns: `(wav_filename, start_time_s, duration_s)`
* A directory containing all the wav files pointed to by `wav_filename` 
* Precomputed mean, invstd .txt files for 64dim mel filterbank features used by the current classifier version on Pod.Cast (see src.params for details)

The TSV file only contains entries for the positive segments in a given `wav_filename`. All the remaining segments in each `wav_filename` are automatically split up into windows assumed as negative by the `AudioFileDataset` class. 

This folder currently contains the following prepared labelled data archives:
* Training datasets:
    * train/WHOIS_PodCast09222019.tar.gz (prepared via Pod.Cast on 09/22/2019)
* Test datasets:
    * test/OS_7_05_2019_08_24_00.tar.gz (labelled by Scott)

> Note if you are contributing data:
> * If you think there's not enough recall in the labels for a wav file, better to chop up and keep only the positive segments as separate wav files. 
> * If you want to add a wavfile containing only negative examples, add one entry in the TSV containing a tiny segment of length 0.01s. `AudioFileDataset` will automatically ignore this as it's too small and process the rest of the file as negative.


# Details on labelled data archives

## train/WHOIS_PodCast_09_22_2019.tar.gz:

This consists of:
* Killer whale ["all cuts"](https://whoicf2.whoi.edu/science/B/whalesounds/fullCuts.cfm?SP=BE7A&YR=-1) scraped from the WHOIS archive. These are ~2-5s clips that contain calls, considered positive examples. There's also some metadata that's scraped and used to join with the tapes below. 
* Unlabelled ["master tapes"](https://whoicf2.whoi.edu/science/B/whalesounds/masterFiles.cfm?MY=-1&SP=BE7A). These are longer unlabelled recordings of an entire session.
    * The master tapes that are referenced in "all cuts" are split into positive/negative examples as above.
    * Remaining unlabelled tapes were labelled with the Pod.Cast system (by a combination of Nithya, Prakruti, Akash, Scott). A preliminary classifier, tuned for high recall (trained on "all cuts" + corresponding master tapes) was used to generate candidates. 

With the first round of annotations added in, this is totally ~6hrs of labelled data i.e. 40% more! 
For a 2.45s window size this is ~8.8k examples with a roughly 60/40 split of negative/positive. 

> Download with this command: `aws --no-sign-request s3 cp s3://acoustic-sandbox/labeled-data/detection/train/WHOIS_PodCast_09_22_2019.tar.gz [DESTINATION]`

## test/OS_SVeirs_07_05_2019_08_24_00.tar.gz:

A test set (do not use for training!) labelled in Audacity by Scott. This is 1/2 hour of WAV data from the Orcasound Lab node on July 05, 2019. 
* This is [this raw data from Scott](https://s3.console.aws.amazon.com/s3/buckets/acoustic-sandbox/labeled-data/classification/killer-whales/southern-residents/20190705/orcasound-lab), put into the standardized format above. The additional "tag" column in the TSV contains exact label ("call","?", etc.), though for now everything is considered as positive. Frequency bands are omitted. (more details on [Orcadata-wiki](wiki/Orcadata-wiki))

> Download with this command: `aws --no-sign-request s3 cp s3://acoustic-sandbox/labeled-data/detection/test/OS_SVeirs_07_05_2019_08_24_00.tar.gz [DESTINATION]`
