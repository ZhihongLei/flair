We adapted this framework (see https://github.com/zalandoresearch/flair) so it supports
    1. Training language models on NLU corpora
        flair.models.language_model.MyLanguageModel
        flair.trainers.language_model_trainer.MyLMTrainer
    2. Jointly trained hybrid NER-LM models 
        flair.models.sequence_tagger_model.HybridSequenceTagger
    3. Beam search for decoding and jointly training the hybrid model
        flair.models.sequence_tagger_model.Beam
        flair.models.sequence_tagger_model.beam_search_batch
        flair.models.sequence_tagger_model.beam_search
        flair.models.sequence_tagger_model.evalute_beam_search
        

We also added helpers for training/evaluationg models
    1. To train models, use
        train.py for original Flair models
        train_lm.py for language models
        train_hybrid.py for jointly trained NER-LM models
    2. To evaluate models, use
        eval.py for original Flair models
        eval_lm.py for language models
        eval_hybrid.py for jointly trained NER-LM models
        beam_search_decoding.py for separately trained NER-LM models
    3. Sample scripts for using these helpers are under scripts/
