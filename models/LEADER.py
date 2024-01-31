# here put the import lib
import torch
import torch.nn as nn
from models.bert_models import TransformerBlock
from models.utils import Contrastive_Loss


class PaddingEncoder(nn.Module):

    def __init__(self, device, emb_dim, prompt_num=None) -> None:
        super().__init__()
        self.device = device
        self.padding_embedding = nn.Embedding(1, emb_dim)

        if not prompt_num:
            self.prompt_num = 1
        else:
            self.prompt_num = prompt_num

        self.med_head, self.proc_head, self.diag_head = nn.Linear(emb_dim, emb_dim*self.prompt_num),\
              nn.Linear(emb_dim, emb_dim*self.prompt_num), nn.Linear(emb_dim, emb_dim*self.prompt_num)


    def forward(self, x):

        prof_emb = self.padding_embedding(torch.zeros(x.shape[0]).long().to(self.device))  # (bs, em_dim)

        return self.diag_head(prof_emb), self.proc_head(prof_emb), self.diag_head(prof_emb)



class ProfileEncoder(nn.Module):

    def __init__(self, device, emb_dim, profile_tokenizer, prompt_num=None) -> None:
        super().__init__()
        self.device = device
        if not prompt_num:
            self.prompt_num = 1
        else:
            self.prompt_num = prompt_num
        self.profile_encoder = nn.ModuleList()

        for tokenizer in profile_tokenizer["word2idx"].values():
            self.profile_encoder.append(nn.Embedding(len(tokenizer), emb_dim))

        self.profile_num = len(profile_tokenizer["word2idx"])
        self.med_head, self.proc_head, self.diag_head = nn.Linear(self.profile_num*emb_dim, emb_dim*self.prompt_num),\
              nn.Linear(self.profile_num*emb_dim, emb_dim*self.prompt_num), nn.Linear(self.profile_num*emb_dim, emb_dim*self.prompt_num)


    def forward(self, x):

        profile_vector = []

        for i in range(self.profile_num):
            profile_vector.append(self.profile_encoder[i](x[:, i]))

        prof_emb = torch.cat(profile_vector, dim=-1)  # (bs, profile_num*emb_dim)

        return self.diag_head(prof_emb), self.proc_head(prof_emb), self.diag_head(prof_emb)



class LEADER(nn.Module):

    def __init__(
        self,
        config,
        args,
        tokenizer,
        device,
        profile_tokenizer=None
    ) -> None:
        
        super().__init__()
        self.vocab_size = len(tokenizer.vocab.word2idx)
        self.med_voc_size = len(tokenizer.med_voc.word2idx)
        self.device = device
        self.emb_dim = args.hidden_size
        self.distill = args.distill
        self.alpha = args.alpha
        self.T = args.temperature
        self.num_trm_layers = args.num_trm_layers
        self.d_loss = args.d_loss
        self.ml_weight = args.ml_weight
        self.align = args.align
        self.align_weight = args.align_weight
        if not args.prompt_num:
            self.prompt_num = 1
        else:
            self.prompt_num = args.prompt_num
        self.embeddings = nn.Embedding(self.vocab_size, self.emb_dim)
        self.dropout = nn.Dropout(p=0.5)

        if args.profile:
            self.profile_encoder = ProfileEncoder(self.device, self.emb_dim, profile_tokenizer, self.prompt_num)
        else:
            self.profile_encoder = PaddingEncoder(self.device, self.emb_dim, self.prompt_num)

        self.med_trm, self.diag_trm, self.proc_trm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.visit_trm = nn.ModuleList()
        self.med_visit_trm, self.diag_visit_trm, self.proc_visit_trm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(self.num_trm_layers):
            self.med_trm.append(TransformerBlock(config))
            self.diag_trm.append(TransformerBlock(config))
            self.proc_trm.append(TransformerBlock(config))
            self.visit_trm.append(TransformerBlock(config))
            self.med_visit_trm.append(TransformerBlock(config))
            self.diag_visit_trm.append(TransformerBlock(config))
            self.proc_visit_trm.append(TransformerBlock(config))

        self.medrec = nn.Sequential(
            nn.ReLU(),
            nn.Linear(3*self.emb_dim, 2*self.emb_dim),
            nn.ReLU(),
            nn.Linear(2*self.emb_dim, self.med_voc_size),
        )
        self.sig = nn.Sigmoid()

        self.loss_fct = nn.BCEWithLogitsLoss(reduction="none")
        self.multi_loss_fct = nn.MultiLabelMarginLoss(reduction="none")

        if self.distill:
            if args.d_loss == "mse":
                self.projector = nn.Linear(2*self.emb_dim, 4096)
                self.distill_loss_fct_mse = nn.MSELoss(reduction="none")
            else:
                return ValueError("Error for distillation loss type.")
        
        if self.align:
            self.align_loss_fct = Contrastive_Loss(project=True, 
                                                   in_dim_1=self.emb_dim,
                                                   in_dim_2=self.emb_dim*self.prompt_num,
                                                   out_dim=self.emb_dim)


    def forward(self, 
                diag_seq, 
                proc_seq, 
                med_seq, 
                seq_mask, 
                labels,
                profile=None,
                multi_label=None,
                llm_output=None,
                **kwargs):

        diag_set_length = torch.sum(diag_seq > 0, dim=2).unsqueeze(-1)  # (bs, max_seq_len, 1)
        diag_set_length[diag_set_length==0] = 1 # avoid divison 0
        proc_set_length = torch.sum(proc_seq > 0, dim=2).unsqueeze(-1)
        proc_set_length[proc_set_length==0] = 1 # avoid divison 0
        med_set_length = torch.sum(med_seq > 0, dim=2).unsqueeze(-1)
        med_set_length[med_set_length==0] = 1 # avoid divison 0
        # mean pooling for diag and proc set
        diag_emb_seq = self.embeddings(diag_seq)    # (bs, max_seq_len, max_set_len, emb_dim)
        proc_emb_seq = self.embeddings(proc_seq)    # (bs, max_seq_len, max_set_len, emb_dim)
        med_emb_seq = self.embeddings(med_seq)    # (bs, max_seq_len, max_set_len, emb_dim)

        # get mask matrics --> (bs, max_seq_len, 1, max_set_len, max_set_len)
        # diag_mask = (diag_seq > 0).view((diag_seq.shape[0], -1)).unsqueeze(1)
        # diag_mask = diag_mask.repeat(1, diag_mask.shape[-1], 1).unsqueeze(1)
        diag_mask = (diag_seq > 0).unsqueeze(2).repeat(1, 1, diag_seq.shape[2], 1).unsqueeze(2)
        proc_mask = (proc_seq > 0).unsqueeze(2).repeat(1, 1, proc_seq.shape[2], 1).unsqueeze(2)
        med_mask = (med_seq > 0).unsqueeze(2).repeat(1, 1, med_seq.shape[2], 1).unsqueeze(2)

        # use the trm to encode the diag / proc / med set
        for i in range(self.num_trm_layers):
            diag_emb_seq = self.diag_trm[i](diag_emb_seq, diag_mask) # (bs, max_seq_len, max_set_len, emb_dim)
            proc_emb_seq = self.proc_trm[i](proc_emb_seq, proc_mask)
            med_emb_seq = self.med_trm[i](med_emb_seq, med_mask)

        # flat the set embedding
        diag_emb_seq = torch.sum(diag_emb_seq * (diag_seq>0).unsqueeze(-1), dim=2) / diag_set_length # (bs, max_seq_len, emb_dim)
        proc_emb_seq = torch.sum(proc_emb_seq * (proc_seq>0).unsqueeze(-1), dim=2) / proc_set_length
        med_emb_seq = torch.sum(med_emb_seq * (med_seq>0).unsqueeze(-1), dim=2) / med_set_length

        # get the profile prompt
        if isinstance(self.profile_encoder, ProfileEncoder):
            diag_pp, proc_pp, med_pp = self.profile_encoder(profile)
        else:
            diag_pp, proc_pp, med_pp = self.profile_encoder(seq_mask)   # (bs, emb_dim)

        # reshape the prompt embedding
        med_pp = med_pp.view(med_pp.shape[0], self.prompt_num, -1)

        # concat the profile prompt
        med_emb_seq = torch.cat([med_pp, med_emb_seq], dim=1)
        med_emb_seq = med_emb_seq[:, :-self.prompt_num, :]    # truncate the last record
        
        # use the trm get the historical representation
        for i in range(self.num_trm_layers):
            diag_emb_seq = self.visit_trm[i](diag_emb_seq, seq_mask.unsqueeze(1).repeat(1, seq_mask.shape[1], 1).unsqueeze(1))
        for i in range(self.num_trm_layers):
            proc_emb_seq = self.visit_trm[i](proc_emb_seq, seq_mask.unsqueeze(1).repeat(1, seq_mask.shape[1], 1).unsqueeze(1))
        for i in range(self.num_trm_layers):
            med_emb_seq = self.visit_trm[i](med_emb_seq, seq_mask.unsqueeze(1).repeat(1, seq_mask.shape[1], 1).unsqueeze(1))

        # avg pooling the visit embedding
        visit_seq_len = torch.sum(seq_mask, dim=1).unsqueeze(1) # seq_mask: (bs, max_seq_len) --> visit_seq_len: (bs, 1)
        diag_emb = torch.sum(diag_emb_seq * seq_mask.unsqueeze(-1), dim=1) / visit_seq_len   # (bs, max_seq_len, em_dim) --> (bs, em_dim)
        proc_emb = torch.sum(proc_emb_seq * seq_mask.unsqueeze(-1), dim=1) / visit_seq_len
        med_emb = torch.sum(med_emb_seq * seq_mask.unsqueeze(-1), dim=1) / visit_seq_len

        output = self.medrec(torch.cat([diag_emb, proc_emb, med_emb], dim=1))

        if self.training:
            loss = self.loss_fct(output, labels).mean(dim=-1)
            if self.align:
                align_loss = self.align_profile(multi_label, med_pp.view(med_pp.shape[0], -1))
                loss += self.align_weight * align_loss.mean(dim=-1)

            if self.distill:
                if self.d_loss == "mse":  # feature-based KD
                    mediator = self.medrec[1](self.medrec[0](torch.cat([diag_emb, proc_emb, med_emb], dim=1)))
                    mediator = self.projector(mediator)
                    pseudo_hidden = llm_output["hidden_states"].float().detach()
                    distill_loss = self.distill_loss_fct_mse(mediator, pseudo_hidden)
                    distill_loss = distill_loss.mean(dim=-1)
                loss = loss + self.alpha * distill_loss

            return loss, 0, output
        
        else:
            return output
        
    def compute_kd(self, y_s, y_t):
        # comptue the distillation loss based on the output of student and teacher
        
        # soften the output from student and teacher
        p_s = torch.sigmoid(y_s / self.T)
        p_t = torch.sigmoid(y_t / self.T)

        # calculate the distill loss
        loss = self.distill_loss_fct_kl(p_s, p_t) * (self.T**2) / y_s.shape[0]
        return loss
    

    def get_loss(self, diag_seq, proc_seq, med_seq, seq_mask, labels, **kwargs):

        loss, _, output = self(diag_seq, proc_seq, med_seq, seq_mask, labels, **kwargs)

        return loss.mean()
    

    def align_profile(self, med_seq, encoder_output):
        # align the output of profile encoder with medication representation
        
        med_seq = med_seq.unsqueeze(1)
        med_seq[med_seq<0] = 0

        med_set_length = torch.sum(med_seq > 0, dim=2).unsqueeze(-1)
        med_set_length[med_set_length==0] = 1 # avoid divison 0
        med_emb_seq = self.embeddings(med_seq)    # (bs, max_seq_len, max_set_len, emb_dim)
        med_mask = (med_seq > 0).unsqueeze(2).repeat(1, 1, med_seq.shape[2], 1).unsqueeze(2)
        for i in range(self.num_trm_layers):
            med_emb_seq = self.med_trm[i](med_emb_seq, med_mask)
        med_emb_seq = torch.sum(med_emb_seq * (med_seq>0).unsqueeze(-1), dim=2) / med_set_length
        loss = self.align_loss_fct(med_emb_seq.squeeze(), encoder_output)
        
        return loss

