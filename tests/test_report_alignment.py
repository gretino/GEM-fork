import torch

from filip.model.losses import report_alignment_loss
from filip.model.report_alignment import ReportAlignmentHead


def test_report_alignment_shapes_and_padding_mask():
    torch.manual_seed(7)
    head = ReportAlignmentHead(6, 5, align_dim=4, topk=2)
    patches = torch.randn(3, 4, 6)
    tokens = torch.randn(3, 5, 5)
    content_mask = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=torch.bool)

    logits, similarities = head(patches, tokens, content_mask)

    assert logits.shape == (3, 3)
    assert similarities.shape == (3, 4, 5)
    assert torch.isfinite(logits).all()


def test_report_alignment_loss_rewards_matching_diagonal():
    aligned = torch.tensor([[8.0, -2.0], [-2.0, 8.0]])
    mismatched = aligned.flip(1)

    assert report_alignment_loss(aligned) < report_alignment_loss(mismatched)


def test_report_alignment_loss_requires_square_batch():
    try:
        report_alignment_loss(torch.zeros(2, 3))
    except ValueError as error:
        assert "square" in str(error)
    else:
        raise AssertionError("Expected non-square logits to be rejected")


def test_prompt_scoring_supports_different_image_and_text_counts():
    torch.manual_seed(11)
    head = ReportAlignmentHead(6, 5, align_dim=4, topk=2)
    patches = torch.randn(2, 4, 6)
    tokens = torch.randn(3, 5, 5)
    content_mask = torch.tensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ], dtype=torch.bool)

    logits, similarities = head.score_prompts(patches, tokens, content_mask)

    assert logits.shape == (2, 3)
    assert similarities.shape == (2, 3, 4, 5)
    assert torch.isfinite(logits).all()
