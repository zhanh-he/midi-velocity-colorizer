import torch
import torch.nn.functional as F

class Recon_Loss:
    def __init__(self, cfg):
        self.cfg = cfg

    def apply_mask(self, est_velo, gt_velo, gt_note):
        # Apply mask based on configuration
        if self.cfg.loss.mask == 'matrix_wise':
            est_velo = est_velo * gt_note
        elif self.cfg.loss.mask == 'element_wise':
            if self.cfg.exp.onset_mask:
                mask = gt_note != 0 # onset element mask
            else:
                mask = gt_velo != 0 # frame element mask
            est_velo, gt_velo = est_velo[mask], gt_velo[mask]
        return est_velo, gt_velo

    def apply_weighting(self, gt_velo):
        # Apply weighting based on velocity values (more weights on edges)
        weight_mask = torch.ones_like(gt_velo)
        lower_bound, upper_bound = 0.424, 0.584
        # lower_bound, upper_bound = 0.47, 0.63
        mid_point = (lower_bound + upper_bound) / 2

        if self.cfg.loss.weight == 'boundary':
            weight_mask[gt_velo < lower_bound] = 1.5
            weight_mask[gt_velo > upper_bound] = 1.5
        elif self.cfg.loss.weight == 'u_shape':
            weight_mask = 1.0 + torch.abs(gt_velo - mid_point) * 3
        elif self.cfg.loss.weight == 'inv_u_shape':
            weight_mask = 1.0 - torch.abs(gt_velo - mid_point) * 3
        return weight_mask

    def which_loss(self, est_velo, gt_velo, weight_mask):
        # Select loss type (BCELoss or MSELoss)
        if self.cfg.loss.type == 'BCELoss':
            return F.binary_cross_entropy(est_velo, gt_velo, weight=weight_mask, reduction='mean')
        elif self.cfg.loss.type == 'MSELoss':
            return F.mse_loss(est_velo * weight_mask, gt_velo * weight_mask, reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {self.cfg.loss.type}")

    def combine_cosim(self, est_velo, gt_velo, recon_loss):
        # Combine with cosine similarity if configured
        if self.cfg.loss.cosim:
            alpha = self.cfg.loss.cosim
            cosim = F.cosine_similarity(est_velo, gt_velo, dim=-1)
            return (1 - alpha) * recon_loss + alpha * (1 - cosim.mean())
        return recon_loss

    def compute(self, est_velo, gt_velo, gt_note):
        # Compute reconstruction loss with optional mask and weighting
        est_velo, gt_velo = self.apply_mask(est_velo, gt_velo, gt_note)
        weight_mask = self.apply_weighting(gt_velo)
        recon_loss = self.which_loss(est_velo, gt_velo, weight_mask)
        return self.combine_cosim(est_velo, gt_velo, recon_loss)


class EvaluationMetrics:
    """Computes MSE, MAE, Recall, and Standard Deviations for velocity prediction."""
    
    def __init__(self, threshold_1=12.7, threshold_2=6.4):
        self.threshold_1 = threshold_1  # Recall threshold (10% tolerance)
        self.threshold_2 = threshold_2  # Stricter recall threshold (5% tolerance)

    def _compute(self, est_velo, gt_velo, gt_onset=None, denormalize=False):
        """Core function for computing evaluation metrics."""
        if gt_onset is not None:
            mask = gt_onset != 0
            est_velo, gt_velo = est_velo[mask], gt_velo[mask]

        if denormalize:  # Only apply denormalization in training mode
            est_velo, gt_velo = (est_velo * 128).round(), (gt_velo * 128).round()
        
        mae = F.l1_loss(est_velo, gt_velo, reduction='mean')
        mse = F.mse_loss(est_velo, gt_velo, reduction='mean')
        sd_ae = torch.std(torch.abs(est_velo - gt_velo))
        recall = cal_recall(est_velo, gt_velo, threshold=self.threshold_1)
        recall_5 = cal_recall(est_velo, gt_velo, threshold=self.threshold_2)
        sd = torch.std(est_velo)
        sd_gt = torch.std(gt_velo)

        return mae, mse, sd_ae, recall, recall_5, sd, sd_gt

    def train(self, est_velo, gt_velo, gt_onset):
        """Training-time evaluation (uses onset masking and denormalization)."""
        return self._compute(est_velo, gt_velo, gt_onset, denormalize=True)

    def interface(self, input_velocities, output_velocities):
        """Interface-based evaluation (compares input/output velocities, without denormalization)."""
        input_tensor, output_tensor = torch.tensor(input_velocities, dtype=torch.float32), torch.tensor(output_velocities, dtype=torch.float32)
        return self._compute(output_tensor, input_tensor, denormalize=False)


# ----------------------- NOTE Additional Evaluation Metrics ------------------------------------------

def cal_precision(est_velo, gt_velo, threshold=0.1):
    diff = torch.abs(est_velo - gt_velo)
    tp = torch.sum(diff < threshold).float()       # True positives
    fp = torch.sum(diff >= threshold).float()      # False positives
    return tp / (tp + fp) if tp + fp > 0 else 0.0  # Avoid division by zero


def cal_recall(est_velo, gt_velo, threshold=0.1):
    diff = torch.abs(est_velo - gt_velo)
    tp = torch.sum(diff < threshold).float()       # True positives
    fn = torch.sum(gt_velo != 0).float() - tp      # False negatives
    return tp / (tp + fn) if tp + fn > 0 else 0.0  # Avoid division by zero


def cal_f1_score(est_velo, gt_velo, threshold=0.1):
    p = cal_precision(est_velo, gt_velo, threshold)
    r = cal_recall(est_velo, gt_velo, threshold)
    return 2 * (p * r) / (p + r) if p + r > 0 else 0.0  # Avoid division by zero