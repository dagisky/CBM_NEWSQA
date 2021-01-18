def make_model(model_config,  N=3,  d_ff=128, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, model_config['input']['symbol_size'], model_config['relational'])
    ff = PositionwiseFeedForward(model_config['input']['symbol_size'], d_ff, dropout)
    position = PositionalEncoding(model_config['input']['symbol_size'], dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(model_config['input']['symbol_size'] , c(attn), c(ff), dropout), N),
        InferenceModule(model_config['input']),
        InputModule(model_config['input']), position)
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model