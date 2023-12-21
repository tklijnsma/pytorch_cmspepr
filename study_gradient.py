import torch
import torch_cmspepr
import torch_cmspepr.objectcondensation as oc

def test_event(verbose=True):

    f = torch.tensor([
        [1.,  10.],
        [2.,  2.],
        [3.,  6.],
        [2.,  4.],
        [2.5, 2.2],
        ], requires_grad=True)
    
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

    if verbose:
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


def L_att(beta, q, x, y, batch):
    """Attraction loss."""

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

        # Number of nodes and number of condensation points in this event
        n = float(row_splits[i_event + 1] - row_splits[i_event])
        n_cond = float(y[left:right].max())

        for i in range(left, right):
            i_cond = cond_point_index[i]

            if i_cond == -1 or i == i_cond:
                # Noise point or condensation point: V_att and V_srp are 0
                pass
            else:
                d = torch.sqrt(torch.sum((x[i] - x[i_cond]) ** 2))
                d_huber = huber(d, 4.0)
                V_att[i] = d_huber * q[i] * q[i_cond] / n

    print(f'{V_att=}')

    return V_att.sum() / float(n_events)




class L_att_manual_backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model_output, y, batch):
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

        V_att = torch.zeros(N, requires_grad=True)

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
                    d = torch.sqrt(torch.sum((x[i] - x[i_cond]) ** 2))
                    d_huber = huber(d+0.00001, 4.0)
                    V_att[i] = d_huber * q[i] * q[i_cond] / n

        ctx.save_for_backward(
            model_output,
            cond_point_index,
            V_att_calculated,
            )

        return V_att.sum() / float(n_events)


    def backward(ctx, grad_output):
        model_output, cond_point_index, V_att_calculated = ctx.saved_tensors

        beta = torch.sigmoid(model_output[:,0])
        q = calc_q(beta)
        x = model_output[:,1:]

        # grad_input = [
        #   [dL/dMO_00 dL/dMO_01 dL/dMO_02],
        #   [dL/dMO_10 dL/dMO_11 dL/dMO_12],
        #   [dL/dMO_20 dL/dMO_21 dL/dMO_22],
        #   [dL/dMO_30 dL/dMO_31 dL/dMO_32],
        #   [dL/dMO_40 dL/dMO_41 dL/dMO_42],
        # ]

        # L = H * q[1] * q[4]
        # q[i] = H * calc_q(beta[i]) = calc_q(sigmoid(model_output[i,0]))

        # L = H * calc_q(sigmoid(model_output_10)) * calc_q(sigmoid(model_output_40))
        # dL/d(model_output_10) = H * q[4] * d_q[beta[1]] * d_sigmoid(model_output_10)
        # dL/d(model_output_40) = H * q[1] * d_q[beta[4]] * d_sigmoid(model_output_40)

        # dL/d(M10) = dH/dM10*Q1*Q4 + H*dQ1/dM10*Q4 + H*Q1*dQ4/dM10
        #           = dH/dM10 * Q1 * Q4
        # dH/dM10 = d_huber(...) * d_sqrt(...) * 2 * (x[1] - x[4])

        grad_input = torch.zeros((5, 3))

        for i in V_att_calculated.nonzero().squeeze(dim=-1):
            j = cond_point_index[i]
            d = torch.sqrt(torch.sum((x[i] - x[j]) ** 2))
            H = huber(d+0.00001, 4.0)
            grad_input[i,0] += H * q[j] * d_q(beta[i]) * d_sigmoid(model_output[i,0]) / 5.
            grad_input[j,0] += H * q[i] * d_q(beta[j]) * d_sigmoid(model_output[j,0]) / 5.
            grad_input[i,1:] += q[i] * q[j] * d_huber(d, 4.) * 1./(2.*d) * 2. * (x[i] - x[j]) / 5.
            grad_input[j,1:] += q[i] * q[j] * d_huber(d, 4.) * 1./(2.*d) * 2. * (x[j] - x[i]) / 5.

        # grad_input[1,0] = q[4] * d_q(beta[1]) * d_sigmoid(model_output[1,0])
        # grad_input[4,0] = q[1] * d_q(beta[4]) * d_sigmoid(model_output[4,0])

        return grad_input, None, None



def test_L_att_grad():
    model_out, feats, weights_autograd, beta, q, x, y, batch = test_event(verbose=True)
    L_autograd = L_att(beta, q, x, y, batch)
    L_autograd.backward()
    print(f'\nweights_autograd.grad=\n{weights_autograd.grad}\n')

    model_out, feats, weights, beta, q, x, y, batch = test_event(verbose=False)

    L = L_att_manual_backward.apply(model_out, y, batch)
    print(f'{L=}  should be {L_autograd}')
    L.backward()

    print(f'{weights.grad[0,0]=}  should be: {weights_autograd.grad[0,0]}')
    print(f'{weights.grad[1,0]=}  should be: {weights_autograd.grad[1,0]}')

    print(f'{weights.grad[0,1]=}  should be: {weights_autograd.grad[0,1]}')
    print(f'{weights.grad[0,2]=}  should be: {weights_autograd.grad[0,2]}')
    print(f'{weights.grad[1,1]=}  should be: {weights_autograd.grad[1,1]}')
    print(f'{weights.grad[1,2]=}  should be: {weights_autograd.grad[1,2]}')

    if torch.allclose(weights.grad, weights_autograd.grad, rtol=0.001):
        print('PASS')
    else:
        print('FAIL')


def L_rep(beta, q, x, y, batch):
    """Repulsion loss."""

    n_events = int(batch.max() + 1)
    N = int(beta.size(0))

    # Translate batch vector into row splits
    row_splits = oc.batch_to_row_splits(batch).type(torch.int)

    y = y.type(torch.int)
    cond_point_index, cond_point_count = oc.cond_point_indices_and_counts(q, y, row_splits)

    V_rep = torch.zeros(N)

    for i_event in range(n_events):
        left = int(row_splits[i_event])
        right = int(row_splits[i_event + 1])

        # Number of nodes and number of condensation points in this event
        n = float(row_splits[i_event + 1] - row_splits[i_event])

        # Indices of the condensation points in this event
        cond_point_indices = left + (
            cond_point_count[left:right] > 0
        ).nonzero().squeeze(dim=-1)

        for i in range(left, right):
            i_cond = cond_point_index[i]
            print(f'{i=} {i_cond=}')
            # V_rep
            for i_cond_other in cond_point_indices:
                if i_cond_other == i_cond: continue  # Don't repulse from own cond point
                d_sq = torch.sum((x[i] - x[i_cond_other]) ** 2)
                V_rep[i] += torch.exp(-4.0 * d_sq) * q[i] * q[i_cond_other] / n

    print(f'{V_rep=}')

    return V_rep.sum() / float(n_events)



class L_rep_manual_backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model_output, y, batch):
        beta = torch.sigmoid(model_output[:,0])
        q = calc_q(beta)
        x = model_output[:,1:]

        n_events = int(batch.max() + 1)
        N = int(beta.size(0))

        # Translate batch vector into row splits
        row_splits = oc.batch_to_row_splits(batch).type(torch.int)

        y = y.type(torch.int)
        cond_point_index, cond_point_count = oc.cond_point_indices_and_counts(q, y, row_splits)

        V_rep = torch.zeros(N, requires_grad=True)

        for i_event in range(n_events):
            left = int(row_splits[i_event])
            right = int(row_splits[i_event + 1])

            # Number of nodes and number of condensation points in this event
            n = float(row_splits[i_event + 1] - row_splits[i_event])

            # Indices of the condensation points in this event
            cond_point_indices = left + (
                cond_point_count[left:right] > 0
            ).nonzero().squeeze(dim=-1)

            for i in range(left, right):
                i_cond = cond_point_index[i]
                print(f'{i=} {i_cond=}')
                # V_rep
                for i_cond_other in cond_point_indices:
                    if i_cond_other == i_cond: continue  # Don't repulse from own cond point
                    d_sq = torch.sum((x[i] - x[i_cond_other]) ** 2)
                    V_rep[i] += torch.exp(-4.0 * d_sq) * q[i] * q[i_cond_other] / n

        ctx.save_for_backward(
            model_output, y, row_splits,
            cond_point_index, cond_point_count
            )

        return V_rep.sum() / float(n_events)


    def backward(ctx, grad_output):
        model_output, y, row_splits, cond_point_index, cond_point_count = ctx.saved_tensors

        n_events = int(len(row_splits)-1)

        beta = torch.sigmoid(model_output[:,0])
        q = calc_q(beta)
        x = model_output[:,1:]

        grad_input = torch.zeros_like(model_output)



        for i_event in range(n_events):
            left = int(row_splits[i_event])
            right = int(row_splits[i_event + 1])

            # Number of nodes and number of condensation points in this event
            n = float(row_splits[i_event + 1] - row_splits[i_event])

            # Indices of the condensation points in this event
            cond_point_indices = left + (
                cond_point_count[left:right] > 0
            ).nonzero().squeeze(dim=-1)

            for i in range(left, right):
                i_cond = cond_point_index[i]
                print(f'{i=} {i_cond=}')
                # V_rep
                for j in cond_point_indices:
                    if j == i_cond: continue  # Don't repulse from own cond point
                    d_sq = torch.sum((x[i] - x[j]) ** 2)
                    grad_input[i,0] += torch.exp(-4.*d_sq) * q[j] * d_q(beta[i]) * d_sigmoid(model_output[i,0]) / n
                    grad_input[j,0] += torch.exp(-4.*d_sq) * q[i] * d_q(beta[j]) * d_sigmoid(model_output[j,0]) / n
                    grad_input[i,1:] += torch.exp(-4.*d_sq) * -8. * (x[i]-x[j]) * q[i] * q[j] / n
                    grad_input[j,1:] += torch.exp(-4.*d_sq) * -8. * (x[j]-x[i]) * q[i] * q[j] / n

        return grad_input, None, None



def test_L_rep_grad():

    model_out, feats, weights_autograd, beta, q, x, y, batch = test_event(verbose=True)
    L_autograd = L_rep(beta, q, x, y, batch)
    L_autograd.backward()
    print(f'\nweights_autograd.grad=\n{weights_autograd.grad}\n')

    model_out, feats, weights, beta, q, x, y, batch = test_event(verbose=False)

    L = L_rep_manual_backward.apply(model_out, y, batch)
    print(f'{L=}  should be {L_autograd}')
    L.backward()

    print(f'{weights.grad[0,0]=}  should be: {weights_autograd.grad[0,0]}')
    print(f'{weights.grad[1,0]=}  should be: {weights_autograd.grad[1,0]}')

    print(f'{weights.grad[0,1]=}  should be: {weights_autograd.grad[0,1]}')
    print(f'{weights.grad[0,2]=}  should be: {weights_autograd.grad[0,2]}')
    print(f'{weights.grad[1,1]=}  should be: {weights_autograd.grad[1,1]}')
    print(f'{weights.grad[1,2]=}  should be: {weights_autograd.grad[1,2]}')

    if torch.allclose(weights.grad, weights_autograd.grad, rtol=0.001):
        print('PASS')
    else:
        print('FAIL')





def L_srp(beta, q, x, y, batch):
    """Short range potential loss."""

    n_events = int(batch.max() + 1)
    N = int(beta.size(0))

    # Translate batch vector into row splits
    row_splits = oc.batch_to_row_splits(batch).type(torch.int)

    y = y.type(torch.int)
    cond_point_index, cond_point_count = oc.cond_point_indices_and_counts(q, y, row_splits)

    V_srp = torch.zeros(N)

    for i_event in range(n_events):
        left = int(row_splits[i_event])
        right = int(row_splits[i_event + 1])

        # Number of nodes and number of condensation points in this event
        n = float(row_splits[i_event + 1] - row_splits[i_event])
        n_cond = float(y[left:right].max())

        # Indices of the condensation points in this event
        cond_point_indices = left + (cond_point_count[left:right] > 0).nonzero().squeeze(dim=-1)

        for i in range(left, right):
            i_cond = cond_point_index[i]
            print(f'{i=} {i_cond=}')
            # V_srp
            d_sq = torch.sum((x[i] - x[i_cond]) ** 2)
            V_srp[i] = (
                -beta[i_cond]
                / (20.0 * d_sq + 1.0)
                / float(cond_point_count[i_cond])  # Number of nodes belonging to cond point
                / n_cond  # Number of condensation points in event
                )

    print(f'{V_srp=}')

    return V_srp.sum() / float(n_events)




class L_srp_manual_backward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model_output, y, batch):
        beta = torch.sigmoid(model_output[:,0])
        q = calc_q(beta)
        x = model_output[:,1:]

        n_events = int(batch.max() + 1)
        N = int(beta.size(0))

        # Translate batch vector into row splits
        row_splits = oc.batch_to_row_splits(batch).type(torch.int)

        y = y.type(torch.int)
        cond_point_index, cond_point_count = oc.cond_point_indices_and_counts(q, y, row_splits)

        V_srp = torch.zeros(N, requires_grad=True)

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

            for i in range(left, right):
                i_cond = cond_point_index[i]
                print(f'{i=} {i_cond=}')
                # V_srp
                d_sq = torch.sum((x[i] - x[i_cond]) ** 2)
                V_srp[i] = (
                    -beta[i_cond]
                    / (20.0 * d_sq + 1.0)
                    / float(cond_point_count[i_cond])  # Number of nodes belonging to cond point
                    / n_cond  # Number of condensation points in event
                    )

        ctx.save_for_backward(
            model_output, y, row_splits,
            cond_point_index, cond_point_count
            )

        return V_srp.sum() / float(n_events)


    def backward(ctx, grad_output):
        model_output, y, row_splits, cond_point_index, cond_point_count = ctx.saved_tensors

        n_events = int(len(row_splits)-1)

        beta = torch.sigmoid(model_output[:,0])
        q = calc_q(beta)
        x = model_output[:,1:]

        grad_input = torch.zeros_like(model_output)

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

            for i in range(left, right):
                j = cond_point_index[i]
                print(f'{i=} {j=}')
                # V_srp
                d_sq = torch.sum((x[i] - x[j]) ** 2)

                grad_input[i,1:] += (
                    -beta[j] / (n_cond * cond_point_count[j])
                    * -1./(20.0 * d_sq + 1.0)**2
                    * 40.*(x[i] - x[j])
                    )
                grad_input[j,1:] += (
                    -beta[j] / (n_cond * cond_point_count[j])
                    * -1./(20.0 * d_sq + 1.0)**2
                    * 40.*(x[j] - x[i])
                    )
                grad_input[j,0] += (
                    -1./(20.0 * d_sq + 1.0)
                    / (n_cond * cond_point_count[j])
                    * d_sigmoid(model_output[j,0])
                    )


        return grad_input, None, None


def test_L_srp_grad():

    model_out, feats, weights_autograd, beta, q, x, y, batch = test_event(verbose=True)
    L_autograd = L_srp(beta, q, x, y, batch)
    L_autograd.backward()
    print(f'\nweights_autograd.grad=\n{weights_autograd.grad}\n')

    model_out, feats, weights, beta, q, x, y, batch = test_event(verbose=False)

    L = L_srp_manual_backward.apply(model_out, y, batch)
    print(f'{L=}  should be {L_autograd}')
    L.backward()

    print(f'{weights.grad[0,0]=}  should be: {weights_autograd.grad[0,0]}')
    print(f'{weights.grad[1,0]=}  should be: {weights_autograd.grad[1,0]}')

    print(f'{weights.grad[0,1]=}  should be: {weights_autograd.grad[0,1]}')
    print(f'{weights.grad[0,2]=}  should be: {weights_autograd.grad[0,2]}')
    print(f'{weights.grad[1,1]=}  should be: {weights_autograd.grad[1,1]}')
    print(f'{weights.grad[1,2]=}  should be: {weights_autograd.grad[1,2]}')

    if torch.allclose(weights.grad, weights_autograd.grad, rtol=0.001):
        print('PASS')
    else:
        print('FAIL')







# test_d_huber()
# test_d_sigmoid()
# test_d_q()
# test_L_att_grad()
# test_L_rep_grad()
test_L_srp_grad()