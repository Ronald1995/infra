"""Source-layout MX scale and low-precision payload decoding for correctness references."""


def dequantize(torch, value, scale, axis=-1, fp4=False):
    axis %= value.ndim
    if fp4:
        # OCP E2M1 packs the earlier element into the low nibble.
        data = value.view(torch.uint8)
        lookup = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6), dtype=torch.float32, device=data.device)
        nibbles = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2)
        payload = lookup[(nibbles & 7).long()] * torch.where(nibbles < 8, 1, -1)
    else:
        payload = value.view(torch.uint8).view(torch.float8_e4m3fn).float()
    exponents = scale.view(torch.uint8).movedim(axis, -2).flatten(-2)
    exponents = exponents.repeat_interleave(32, dim=-1)[..., :payload.shape[axis]]
    return torch.ldexp(payload.movedim(axis, -1), exponents.int() - 127).movedim(-1, axis)
