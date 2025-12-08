class Architecture:
    def __init__(self, blocks, model_dir, num_classes, batch_size=100):
        self.blocks = blocks  # sequence of blocks (e.g., [Vector(), Fourier1D(), ...])
        self.model_dir = model_dir  # base directory for saving weights / logs per block & layer
        self.num_classes = num_classes  # total number of classes (used by blocks)
        self.batch_size = batch_size  # optional mini-batch size hint (e.g., for covariance batching)

    def __call__(self, Z, y=None):
        for b, block in enumerate(self.blocks):  # iterate through blocks in order (feed-forward pipeline)
            block.load_arch(self, b)  # give the block a handle to this arch (paths, num_classes, id)
            self.init_loss()  # reset loss containers for THIS block (per-block curves)

            Z = block.preprocess(Z)  # block-specific input normalization / reshaping
            Z = block(Z, y)  # run the block; may construct or load E, C_j, γ and update losses
            Z = block.postprocess(Z)  # block-specific output normalization before passing to next block
        return Z  # final features after all blocks

    def __getitem__(self, i):
        return self.blocks[i]  # allow indexing: arch[i] -> i-th block

    def init_loss(self):
        # fresh loss dict (per block): each block fills these across its internal layers
        self.loss_dict = {"loss_total": [],  # ΔR = R - Rc (the objective to maximize)
                          "loss_expd": [],  # R term (overall coding rate)
                          "loss_comp": []}  # Rc term (class-wise coding rate)

    def update_loss(self, layer, loss_total, loss_expd, loss_comp):
        # append per-layer numbers (called by block after each layer forward step)
        self.loss_dict["loss_total"].append(loss_total)
        self.loss_dict["loss_expd"].append(loss_expd)
        self.loss_dict["loss_comp"].append(loss_comp)
        # quick console readout for monitoring training dynamics
        print(f"layer: {layer} | loss_total: {loss_total:5f} | loss_expd: {loss_expd:5f} | loss_comp: {loss_comp:5f}")