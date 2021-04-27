# CNN/DailyMail Summarization with TNLRv3

Jon Sleep 2/1/2021, updated 4/5

This is an example explaining entire pipeline for the CNN/DailyMail Summarization task using the TNLRv3 seq2seq model.

[Source code location](https://o365exchange.visualstudio.com/O365%20Core/_git/ELR?path=%2Fsources%2Fdev%2FSubstrateInferences%2Fmarlin_Scenarios%2Ftnlrv3_seq2seq&version=GBu%2Felr%2Frefactor&_a=contents).

Dataset used: CNN/DailyMail
Model : TNLRv3/UniLMv2 - [paper](https://arxiv.org/abs/2002.12804)

We start with a pretrained checkpoint from Turing team and finetune on CNN/Dailymail dataset.


## Run in AzureML

Go to the jupyter notebook in the scenario folder to see how to run in AzureML [here](https://o365exchange.visualstudio.com/O365%20Core/_git/ELR?path=%2Fsources%2Fdev%2FSubstrateInferences%2FMarlin_Scenarios%2Ftnlrv3_seq2seq&version=GBmaster&_a=contents)

## Run Locally

### Download Data

Download CNN/DailyMail uncased & tokenized from [here](https://aka.ms/alexanderrepo) and put into D:/data/cnndm_ft

### Preprocess and analyze

    python data_finetune.py --data.data_dir 'D:/data/cnndm_ft' --config_path finetune_config_local.yaml

### Train

Alter finetune_config_local.yaml with different hyperparams as desired

    python train.py --data.data_dir D:/data/cnndm_ft --config_path finetune_config_local.yaml

### Prepare data for decode
Copy unfeaturized files from cnndm_ft and:

    python data_decode.py -data.data_dir 'D:/data/cnndm_decode' --config_path decode_config_local.yaml
### Decode inference

    python decode.py --module.model_dir checkpoints --config_path decode_config_local.yaml --module.output_dir D:/data/cnndm_outputs

Should get results from source to summary like:

Source (tokenized):

``` on the 6th of april 1996 , san jose clash and dc united strode out in front of 31 , 68 ##3 expect ##ant fans at the spartan stadium in san jose , california . the historic occasion was the first ever major league soccer match - - a brave new dawn for the world ' s favorite sport in a land its charms had yet to conquer . sum ##mar ##izing the action for espn , commentator ty ke ##ough eagerly described the moment ##ous " birth of a new era for american soccer . " looking back at footage from that bal ##my evening now it ' s hard not to feel a certain nostalgia . bag ##gy shirts , questionable hairs ##tyle ##s and strange rule adaptations to make games more exciting were all part of the format ##ive mls experience . countdown clocks were employed to provide drama at the end of each half . even more bizarre ##ly , tied games were settled by shootout ##s that saw attacking players run with the ball from 35 - yards out before attempting to beat the opposing goalkeeper . as the mls prepares to mark the beginning of its 20th season , it ' s hard to comprehend just how much the league has progressed in the intervening period . long gone is the desire to tam ##per with the rules of the game for a start . attendance ##s are higher than ever before while the number of teams involved has doubled from 10 in the 1996 campaign to 20 in 2015 . a further four are set to be added by 2020 . on top of this , the new season is the first of a new domestic tv and media rights deal with fox , espn and un ##ivision worth $ 700 million over eight years . this figure may pale beside the $ 5 . 1 billion recently paid by uk broadcasters for the english premier league , the richest football league in the world , but it represents a trip ##ling in value of the previous mls deal . according to phil raw ##lins , co - primary owner and president of the new mls franchise , orlando city soccer club , " the industry and the game itself has moved on dramatically " in the u . s . . he believes what would equal 50 years growth in most other industries has been experienced in the first two decades of the mls . raw ##lins ' club is a prime example of this rapid transformation . he describes players being pushed out of changing facilities because of a schedule clash with a yoga class not so long ago . this weekend 60 , 000 fans are expected to witness orlando city ' s opening weekend fixture against new york city , another new club making their mls bow . world cup winners ka ##ka and david villa will turn out for orlando and new york city respectively . " we ' re just on the crest of the wave at the moment , " raw ##lins said of football ' s american progress . " can it be the number two , number three sport in this country ? yes , i think it can . and it can be in a short space of time . " these positive assertion ##s are backed by the huge interest u . s . fans showed in last year ' s world cup in brazil . team usa ' s group stage clash with portugal attracted 25 million ```
    
Summary (tokenized): 

``` orlando city ' s opening weekend match against new york city on saturday . [X_SEP] attendances are higher than ever before while the number of teams involved has doubled from 10 in the 1996 campaign to 20 in 2015 . [X_SEP] the new season is the first of a new domestic tv and media rights deal with fox , espn and univision . ```