# Repo to explain FastAI modeling methodology
- Author : Aayush Agrawal (aaagraw@microsoft.com)

# Model data 
The base data used here is hosted on Current test set for evaluation is hosted on [Orca Sound website ](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#test-sets). In specific the model was  trained with the following dataset -

- [WHOIS09222019_PodCastRound1](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#WHOIS_PodCastRound1) (~6hrs, open data source – Orca call around the planet, Good for generic models)
- [OrcasoundLab07052019_PodCastRound2](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab07052019_PodCastRound2) (~1.2hrs, live hydrophone data - SRKW)
- [OrcasoundLab09272017_PodCastRound3](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab09272017_PodCastRound3) (~1.6hrs, live hydrophone data - SRKW)

And testing in evaluation section of FastAI start script is done on this data - 
- [OrcasoundLab07052019_Test](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab07052019_Test) (~30min, very call-rich, good SNR conditions)
- [OrcasoundLab09272017_Test](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab07052019_Test) (~21min, more typical distribution, good SNR conditions)

# Data Directory structure in this folder(not uploaded on Git) -
```
fastai/data
	- test
	- train
		- train_data_09222019
            - wav
            - train.tsv
        - Round2_OS_07_05
            - wav
            - train.tsv
        - Round3_OS_09_27_2017
            - wav
            - train.tsv
        - mldata
            - all
                - model
                - negative
                - positive

    - test
        - all
        - OrcasoundLab07052019_Test
            - test2Sec [Created later]
            - wav
            - test.tsv
        - OrcasoundLab09272017_Test
            - wav
            - test.csv
    
	
```   
