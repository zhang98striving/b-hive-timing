import torch


class Attacks:
    def __init__(
        self,
        device=torch.device("cpu"),
        integer_positions=None,
        default_values=None,
        epsilon=0.1,
        epsilon_factors=True,
        iterations=1,
        reduce=True,
        restrict_impact=-1,
        **kwargs,
    ):
        super(Attacks, self).__init__(**kwargs)

        self.device = device
        self.epsilon = epsilon
        if epsilon_factors:
            print(
                "Individual epsilons per feature not yet implemented. Using epsilon=1 for all variables instead."
            )
            self.epsilons_per_feature = [1.0, 1.0, 1.0, 1.0]
        else:
            self.epsilons_per_feature = [1.0, 1.0, 1.0, 1.0]
        self.iterations = iterations
        self.reduce = reduce
        self.restrict_impact = restrict_impact

        self.torch_zero = torch.tensor(0.0).to(self.device)
        self.integers = [integer.to(self.device) for integer in integer_positions]
        self.defaults = [default.to(self.device) for default in default_values]

    def nominal(self, inputs, truth, criterion, model):
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

    def pgd(self, inputs, truth, criterion, model):
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

        with torch.no_grad():
            if self.restrict_impact > 0:
                for index, (input, adversarial_input) in enumerate(
                    zip(inputs, adversarial_inputs)
                ):
                    adversarial_input[index] = torch.clamp(
                        adversarial_input[index],
                        min=input - self.restrict_impact * torch.abs(input),
                        max=input + self.restrict_impact * torch.abs(input),
                    )
        return *[input.detach() for input in adversarial_inputs], truth

    # Code adapted from https://github.com/LTS4/DeepFool, based on https://arxiv.org/pdf/1511.04599.pdf
    def jetfool(self, inputs, truth, criterion, model):
        print("JetFool attack not yet implemented. Returning nominal inputs.")
        return *inputs, truth
