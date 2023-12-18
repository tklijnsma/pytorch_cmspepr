import torch
import torch_cmspepr
import torch_cmspepr.objectcondensation as oc

def test_event():

    f = torch.tensor([
        [1.,  10.],
        [2.,  2.],
        [3.,  6.],
        [2.,  4.],
        [2.5, 2.2],
        ], requires_grad=False)
    
    w = torch.tensor([
        [ 1.1, 1., 1.],
        [-1.0, 1., 1.],
        ], requires_grad=True)

    model_out = f.matmul(w)
    model_out_exp = torch.tensor([          # y
        [-8.9000, 11.0000, 11.0000], # 0
        [ 0.2000,  4.0000,  4.0000], # 1
        [-2.7000,  9.0000,  9.0000], # 0
        [-1.8000,  6.0000,  6.0000], # 0
        [ 0.5500,  4.7000,  4.7000]  # 1 <-- cond point
        ], requires_grad=False)
    assert torch.allclose(model_out, model_out_exp)

    x = model_out[:,1:]
    y = torch.LongTensor([0, 1, 0, 0, 1])
    batch = torch.LongTensor([0, 0, 0, 0, 0])

    beta = torch.sigmoid(model_out[:, 0])
    q = calc_q(beta)


    print('In test_event:')
    print(f'{f=}')
    print(f'{w=}')
    print(f'{x=}')
    print(f'{beta=}')
    print(f'{q=}')
    print('')

    return model_out, f, w, beta, q, x, y, batch


def calc_q(beta: torch.Tensor, qmin=1.0) -> torch.Tensor:
    """
    Calculate the value of q.

    Args:
        beta (torch.Tensor): The input beta tensor.
        qmin (float, optional): The minimum value of q. Defaults to 1.0.

    Returns:
        torch.Tensor: The calculated value of q.
    """
    return (beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + qmin


def d_q(beta: torch.Tensor) -> torch.Tensor:
    """Derivative of q.

    Args:
        beta (torch.Tensor): Beta tensor.

    Returns:
        torch.Tensor: Derivative of q.
    """
    # q(beta) = (beta.clip(0.0, 1 - 1e-4) / 1.002).arctanh() ** 2 + qmin
    a = 0.0
    b = 1 - 1e-4
    c = 1.002
    qmin = 1.0
    if beta <= a or beta >= b: return torch.tensor(0.)
    return (
        2. / c * (beta.clip(a, b) / c).arctanh()
        * 1./(1.-(beta.clip(a, b) / c)**2)
        )

def test_d_q():
    """Test derivative of q."""
    for num in [0.0, 0.1, 0.5, 0.9, 1.0]:
        print(f'\n{num=}')
        beta = torch.tensor(num, requires_grad=True)
        q = oc.calc_q_betaclip(beta)
        q.backward()
        print(f'{q=}')
        print(f'{beta.grad=}')
        print(f'{d_q(beta)=}')
        assert torch.allclose(beta.grad, d_q(beta))


def huber(x, delta):
    return torch.where(
        torch.abs(x) < delta,
        x**2,
        2 * delta * (torch.abs(x) - delta),
        )

def d_huber(x, delta):
    return torch.where(
        torch.abs(x) < delta,
        2. * x,
        2. * delta * torch.sign(x),
        )


def d_sigmoid(x):
    return torch.exp(-x) / (1 + torch.exp(-x))**2


def test_d_sigmoid():
    for num in [-4., -.2, 0., .2, 4.]:
        print(f'\n{num=}')
        x = torch.tensor(num, requires_grad=True)
        y = torch.sigmoid(x)
        d_y = d_sigmoid(x)
        print(f'{y=}')
        y.backward()
        print(f'{x.grad=}')
        print(f'{d_y=}')
        assert torch.allclose(x.grad, d_y)


def test_d_huber():
    for num in [-100., -.2, 0., .2, 100.]:
        print(f'\n{num=}')
        x = torch.tensor(num, requires_grad=True)
        delta = 4.0
        y = huber(x, delta)
        d_y = d_huber(x, delta)
        print(f'{y=}')
        y.backward()
        print(f'{x.grad=}')
        print(f'{d_y=}')
        assert torch.allclose(x.grad, d_y)


# @torch.jit.script
def L_att(beta, q, x, y, batch):
    """Simplified attraction loss: Just do q_i * q_j now, for any node i that has a
    condensation point j.
    """

    n_events = int(batch.max() + 1)
    N = int(beta.size(0))

    # Translate batch vector into row splits
    row_splits = oc.batch_to_row_splits(batch).type(torch.int)

    y = y.type(torch.int)
    cond_point_index, cond_point_count = oc.cond_point_indices_and_counts(q, y, row_splits)

    V_att = torch.zeros(N)

    for i_event in range(n_events):
        left = int(row_splits[i_event])
        right = int(row_splits[i_event + 1])

        # # Number of nodes and number of condensation points in this event
        # n = float(row_splits[i_event + 1] - row_splits[i_event])
        # n_cond = float(y[left:right].max())

        for i in range(left, right):
            i_cond = cond_point_index[i]

            if i_cond == -1 or i == i_cond:
                # Noise point or condensation point: V_att and V_srp are 0
                pass
            else:
                # d_sq = torch.sum((x[i] - x[i_cond]) ** 2)
                # d = torch.sqrt(d_sq)
                # d_plus_eps = d + 0.00001
                # d_huber = (
                #     d_plus_eps**2
                #     if d_plus_eps <= 4.0
                #     else 2.0 * 4.0 * (d_plus_eps - 4.0)
                # )
                # V_att[i] = d_huber * q[i] * q[i_cond] / n
                V_att[i] = q[i] * q[i_cond]

    print(f'{V_att=}')

    return V_att.sum() / float(n_events)



def manual_gradient_calculation():
    """Manual calculation of the gradients of w00 and w10.
    """

    model_out, feats, weights, beta, q, x, y, batch = test_event()

    f0 = feats[:,0].squeeze()
    f1 = feats[:,1].squeeze()

    x0 = x[:,0]
    x1 = x[:,1]

    H = huber( torch.sqrt((x0[1]-x0[4])**2 + (x1[1]-x1[4])**2) + 0.00001 , 4.)


    w00 = torch.tensor(1.1, requires_grad=True)
    w10 = torch.tensor(-1.0, requires_grad=True)

    f0_1 = torch.tensor(2.0)
    f1_1 = torch.tensor(2.0)
    f0_4 = torch.tensor(2.5)
    f1_4 = torch.tensor(2.2)

    x0_1 = w00*f0_1 + w10*f1_1
    x0_4 = w00*f0_4 + w10*f1_4

    beta_1 = torch.sigmoid(w00*f0_1+w10*f1_1)
    beta_4 = torch.sigmoid(w00*f0_4+w10*f1_4)

    q_1 = calc_q(beta_1)
    q_4 = calc_q(beta_4)

    L = q_1 * q_4
    L.backward()

    print(f'{x0_1=} {x0_4=}')
    print(f'{beta_1=} {beta_4=}')
    print(f'{q_1=} {q_4=}')
    print(f'{L=}')

    dw00 = (
        q_4 * d_q(beta_1) * d_sigmoid(w00*f0_1+w10*f1_1) * f0_1
        +
        q_1 * d_q(beta_4) * d_sigmoid(w00*f0_4+w10*f1_4) * f0_4
        )

    dw10 = (
        q_4 * d_q(beta_1) * d_sigmoid(w00*f0_1+w10*f1_1) * f1_1
        +
        q_1 * d_q(beta_4) * d_sigmoid(w00*f0_4+w10*f1_4) * f1_4
        )

    print(f'{w00.grad=}  {dw00=};  ratio={dw00/w00.grad}')
    print(f'{w10.grad=}  {dw10=};  ratio={dw10/w10.grad}')





class L_att_manual_backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model_output, y, batch):
        # ctx.save_for_backward(x0, beta, q, x, y)
        
        beta = torch.sigmoid(model_output[:,0])
        q = calc_q(beta)
        x = model_output[:,1:]

        n_events = int(batch.max() + 1)
        N = int(beta.size(0))

        # Translate batch vector into row splits
        row_splits = oc.batch_to_row_splits(batch).type(torch.int)

        y = y.type(torch.int)
        cond_point_index, cond_point_count = oc.cond_point_indices_and_counts(q, y, row_splits)

        is_noise = y == 0

        V_att = torch.zeros(N)

        for i_event in range(n_events):
            left = int(row_splits[i_event])
            right = int(row_splits[i_event + 1])

            # Number of nodes and number of condensation points in this event
            n = float(row_splits[i_event + 1] - row_splits[i_event])
            n_cond = float(y[left:right].max())

            # Indices of the condensation points in this event
            cond_point_indices = left + (
                cond_point_count[left:right] > 0
            ).nonzero().squeeze(dim=-1)
            # Indices of the noise nodes in this event
            noise_indices = left + is_noise[left:right].nonzero().squeeze(dim=-1)

            V_att_calculated = torch.zeros(N, dtype=torch.bool)

            for i in range(left, right):
                i_cond = cond_point_index[i]

                if i_cond == -1 or i == i_cond:
                    # Noise point or condensation point: V_att and V_srp are 0
                    pass
                else:
                    V_att_calculated[i] = True
                    d_sq = torch.sum((x[i] - x[i_cond]) ** 2)
                    d = torch.sqrt(d_sq)
                    d_plus_eps = d + 0.00001
                    d_huber = (
                        d_plus_eps**2
                        if d_plus_eps <= 4.0
                        else 2.0 * 4.0 * (d_plus_eps - 4.0)
                    )
                    # V_att[i] = d_huber * q[i] * q[i_cond] / n
                    V_att[i] = q[i] * q[i_cond]

        ctx.save_for_backward(
            model_output,
            cond_point_index,
            V_att_calculated,
            )

        print(f'{V_att=}')

        return V_att.sum() / float(n_events)


    def backward(ctx, grad_output):
        model_output, cond_point_index, V_att_calculated = ctx.saved_tensors
        
        beta = torch.sigmoid(model_output[:,0])
        q = calc_q(beta)
        # x = model_output[:,1:]

        print(f'{grad_output=}')

        grad_input = torch.zeros((5, 3))


        f0_1 = torch.tensor(2.0)
        f1_1 = torch.tensor(2.0)
        f0_4 = torch.tensor(2.5)
        f1_4 = torch.tensor(2.2)


        grad_00 = torch.tensor(0.)
        for i in torch.nonzero(V_att_calculated).squeeze(-1):
            j = cond_point_index[i]
            grad_00 += q[i] * d_q(beta[j]) * d_sigmoid(model_output[j,0]) # * f0_4
            grad_00 += q[j] * d_q(beta[i]) * d_sigmoid(model_output[i,0]) # * f0_1

        grad_10 = torch.tensor(0.)
        for i in torch.nonzero(V_att_calculated).squeeze(-1):
            j = cond_point_index[i]
            grad_10 += q[i] * d_q(beta[j]) * d_sigmoid(model_output[j,0]) # * f1_4
            grad_10 += q[j] * d_q(beta[i]) * d_sigmoid(model_output[i,0]) # * f1_1


        print(f'{grad_00=}  {grad_10=}')

        # grad_input[:,0] = grad_00
        # grad_input[:,1] = grad_10

        grad_input[1,0] = grad_00

        return grad_input, None, None



def use_class():
    model_out, feats, weights, beta, q, x, y, batch = test_event()

    L = L_att_manual_backward.apply(model_out, y, batch)
    L.backward()

    print(f'{L=}  should be 2.1486')

    print(f'{weights.grad[0,0]=}  should be: 3.3460')
    print(f'{weights.grad[1,0]=}  should be: 3.1073')



# test_d_huber()
# test_d_sigmoid()
# test_d_q()

manual_gradient_calculation()
# use_class()