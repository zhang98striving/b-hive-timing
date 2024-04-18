import torch


class Attacks:
    def __init__(
        self,
        number_classes,
        device=torch.device("cpu"),
        integer_positions=None,
        default_values=None,
        epsilon=0.0,
        epsilon_factors=True,
        iterations=1,
        reduce=True,
        restrict_impact=-1,
        overshoot=0.02,
        **kwargs,
    ):
        super(Attacks, self).__init__(**kwargs)
        self.device = device
        self.torch_zero = torch.tensor(0.0).to(self.device)
        self.torch_one = torch.tensor(1.0).to(self.device)
        self.torch_inf = torch.tensor(float("inf")).to(device)
        self.number_classes = number_classes
        self.epsilon = epsilon
        if epsilon_factors:
            print(
                "Individual epsilons per feature are hardcoded for DeepJet at the moment. Turn them off, if you use a different tagger."
            )
            self.epsilons_per_feature = [
                self.torch_one,
                self.torch_one,
                self.torch_one,
                self.torch_one,
            ]
        else:
            self.epsilons_per_feature = [
                self.torch_one,
                self.torch_one,
                self.torch_one,
                self.torch_one,
            ]
        self.iterations = iterations
        self.reduce = reduce
        self.restrict_impact = restrict_impact
        self.integers = [integer.to(self.device) for integer in integer_positions]
        self.defaults = [default.to(self.device) for default in default_values]
        self.overshoot = overshoot

    def nominal(self, inputs, truth, model, criterion):
        return *inputs, truth

    def do_not_change(self, inputs, adversarial_vectors):
        if self.reduce == False:
            return adversarial_vectors

        elif self.reduce == True:
            masks = []
            for input, integer, default in zip(inputs, self.integers, self.defaults):
                mask = input == default
                mask[..., integer] = True
                masks.append(mask)

            for index, (mask, adversarial_vector) in enumerate(
                zip(masks, adversarial_vectors)
            ):
                adversarial_vectors[index] = torch.where(
                    mask, self.torch_zero, adversarial_vector
                )
            return adversarial_vectors

    def already_fooled(self, adversarial_vectors, nominal_labels, adversarial_labels):
        initial_mask = nominal_labels == adversarial_labels

        for index, adversarial_vector in enumerate(adversarial_vectors):
            shape = list(adversarial_vector.shape)
            shape[0] = -1
            mask = initial_mask.clone().reshape(-1, *[1 for i in range(len(shape) - 1)])
            mask = mask.expand(shape)
            adversarial_vectors[index] = torch.where(
                mask, adversarial_vector, self.torch_zero
            )
        return adversarial_vectors

    def limit_relative_change(self, inputs, adversarial_inputs):
        with torch.no_grad():
            if self.restrict_impact > 0:
                for index, (input, adversarial_input) in enumerate(
                    zip(inputs, adversarial_inputs)
                ):
                    adversarial_inputs[index] = torch.clamp(
                        adversarial_input,
                        min=input - self.restrict_impact * torch.abs(input),
                        max=input + self.restrict_impact * torch.abs(input),
                    )
        return adversarial_inputs

    def pgd(self, inputs, truth, model, criterion):
        alphas = []
        for e in self.epsilons_per_feature:
            alphas.append(self.epsilon * e / self.iterations)

        adversarial_inputs = []
        for input in inputs:
            adversarial_inputs.append(
                input.clone().detach().to(self.device).requires_grad_(True)
            )

        for i in range(self.iterations):
            prediction = model.forward(*adversarial_inputs)

            loss = criterion(prediction, truth).mean()

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                gradients = []
                for input in adversarial_inputs:
                    gradients.append(input.grad.detach().sign())

                deltas = []
                for input, alpha, gradient in zip(
                    adversarial_inputs, alphas, gradients
                ):
                    deltas.append(
                        torch.clamp(
                            (input + alpha * gradient) - input,
                            min=-alpha * self.iterations,
                            max=alpha * self.iterations,
                        )
                    )

                deltas = self.do_not_change(inputs, deltas)

                for index, delta in enumerate(deltas):
                    adversarial_inputs[index] += delta

        adversarial_inputs = self.limit_relative_change(inputs, adversarial_inputs)

        return *[input.detach() for input in adversarial_inputs], truth

    # Code adapted from https://github.com/LTS4/DeepFool, based on https://arxiv.org/pdf/1511.04599.pdf
    def jetfool(self, inputs, truth, model, criterion):

        batch_size = inputs[0].shape[0]

        prediction = model.forward(*inputs)
        I = torch.argsort(prediction, dim=1, descending=True)
        label = I[:, 0]

        adversarial_inputs = []
        ws = []
        r_tots = []
        for input in inputs:
            adversarial_inputs.append(
                input.clone().detach().to(self.device).requires_grad_(True)
            )
            shape = input.shape
            ws.append(torch.zeros(shape).to(self.device))
            r_tots.append(torch.zeros(shape).to(self.device))

        loop = 0

        fs = model.forward(*adversarial_inputs)

        k_i = label

        while torch.any(k_i == label).item() and loop < self.iterations:
            perturbations = [self.torch_inf for i in range(len(inputs))]

            fs_sorted = torch.gather(fs, 1, I)
            fs_sorted[:, 0].backward(
                gradient=torch.ones(batch_size).to(self.device), retain_graph=True
            )

            original_gradients = []
            for input in adversarial_inputs:
                original_gradients.append(input.grad.clone())

            for c in range(1, self.number_classes):
                for input in adversarial_inputs:
                    input.grad.zero_()

                fs_sorted[:, c].backward(
                    gradient=torch.ones(batch_size).to(self.device), retain_graph=True
                )

                adversarial_gradients = []
                for input in adversarial_inputs:
                    adversarial_gradients.append(input.grad.clone())

                w_cs = []
                for original_gradient, adversarial_gradient in zip(
                    original_gradients, adversarial_gradients
                ):
                    w_cs.append((adversarial_gradient - original_gradient).detach())

                f_c = (fs_sorted[:, c] - fs_sorted[:, 0]).detach()

                new_perturbations = []
                for w_c in w_cs:
                    new_perturbation = torch.abs(f_c) / torch.linalg.norm(
                        w_c.reshape(batch_size, -1), axis=1
                    )
                    new_perturbations.append(
                        torch.where(
                            new_perturbation == self.torch_inf,
                            self.torch_zero,
                            new_perturbation,
                        )
                    )

                for index, (perturbation, new_perturbation, w_c, w) in enumerate(
                    zip(perturbations, new_perturbations, w_cs, ws)
                ):
                    compare = new_perturbation < perturbation
                    perturbations[index] = torch.where(
                        compare, new_perturbation, perturbation
                    )
                    w[compare, ...] = w_c[compare, ...]
                    ws[index] = w

            r_is = []
            for w, perturbation in zip(ws, perturbations):
                w_dimension = w.ndim
                shape = [-1]
                for i in range(w_dimension - 1):
                    shape.append(1)

                r_is.append(
                    torch.nan_to_num(
                        w
                        * (
                            (perturbation + 1e-4)
                            / (torch.linalg.norm(w.reshape(batch_size, -1), axis=1))
                        ).reshape(*shape),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                )

            r_is = self.already_fooled(r_is, label, k_i)

            for index, r_i in enumerate(r_is):
                r_tots[index] += r_i

            r_tots = self.do_not_change(inputs, r_tots)

            for index, (nominal_input, r_tot) in enumerate(zip(inputs, r_tots)):
                adversarial_inputs[index] = (
                    nominal_input + (1 + self.overshoot) * r_tot * self.epsilon
                ).requires_grad_(True)

            fs = model.forward(*adversarial_inputs)
            k_i = torch.argmax(fs, axis=1)

            loop += 1

        for index, r_tot in enumerate(r_tots):
            r_tots[index] *= 1 + self.overshoot

        adversarial_inputs = self.limit_relative_change(inputs, adversarial_inputs)

        return *[input.detach() for input in adversarial_inputs], truth
