import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]   


def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions  # 计算单位方向上的球谐波
    using hardcoded SH polynomials.  # 使用硬编码的SH多项式
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported   球谐函数的度数l
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]   球谐系数
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0  # 断言确保球谐系数的阶数 deg 在合理的范围内，level在0-4之间
    # 勒让德多项式(Legendre Polynomials)里的基底函数分为几组，每组有一个index(level)，其中index(level)越高，频率越高，
    # 每个组不是单独的一个基底，第l组有（l+1）² - 1个基地函数组成，用y（l,m）代表第l组里面的第m个基底，所有的基底都是正交的
    coeff = (deg + 1) ** 2  # 计算给定阶数下的球谐系数的总数
    assert sh.shape[-1] >= coeff  # 确保提供的球谐系数张量 sh 至少具有足够的长度来容纳所需的球谐系数数量，比最多系数的一组的数目大就可以。

    result = C0 * sh[..., 0]  # 计算球谐系数展开的第一个系数。对于阶数为 0 的情况，只计算了一个球谐系数
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                C1 * y * sh[..., 1] +
                C1 * z * sh[..., 2] -
                C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                    C2[0] * xy * sh[..., 4] +
                    C2[1] * yz * sh[..., 5] +
                    C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                    C2[3] * xz * sh[..., 7] +
                    C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                C3[1] * xy * z * sh[..., 10] +
                C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                C3[5] * z * (xx - yy) * sh[..., 14] +
                C3[6] * x * (xx - 3 * yy) * sh[..., 15])

                if deg > 3:
                    result = (result + C4[0] * xy * (xx - yy) * sh[..., 16] +
                            C4[1] * yz * (3 * xx - yy) * sh[..., 17] +
                            C4[2] * xy * (7 * zz - 1) * sh[..., 18] +
                            C4[3] * yz * (7 * zz - 3) * sh[..., 19] +
                            C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20] +
                            C4[5] * xz * (7 * zz - 3) * sh[..., 21] +
                            C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22] +
                            C4[7] * xz * (xx - 3 * yy) * sh[..., 23] +
                            C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy)) * sh[..., 24])
    return result


# 将RGB值转为球谐系数值
def RGB2SH(rgb):
    # 将输入的 RGB 值减去 0.5，这样做是为了将 RGB 值从 [0, 1] 的范围映射到 [-0.5, 0.5] 的范围
    return (rgb - 0.5) / C0


# 将球谐系数值转为RGB值
def SH2RGB(sh):
    return sh * C0 + 0.5
