import torch


class Attacks:
    def __init__(self, device=torch.device("cpu"), epsilon=0.1, epsilon_factors=True, iterations=1, reduce=True, restrict_impact=-1, **kwargs):
        super(Attacks, self).__init__(**kwargs)

        self.device = device
        self.epsilon = epsilon
        if epsilon_factors:
            print("Individual epsilons per feature not yet implemented. Using epsilon=1 for all variables instead.")
            self.epsilons_per_feature = [1.0, 1.0, 1.0, 1.0]
        else:
            self.epsilons_per_feature = [1.0, 1.0, 1.0, 1.0]
        self.iterations = iterations
        self.reduce = reduce
        self.restrict_impact = restrict_impact

        self.torch_zero = torch.tensor(0.0).to(self.device)
        self.glob_int = torch.tensor([2, 3, 4, 5, 8, 13, 14]).to(self.device)
        self.cpf_int = torch.tensor([12, 13, 14, 15]).to(self.device)
        self.npf_int = torch.tensor([2]).to(self.device)
        self.vtx_int = torch.tensor([3]).to(self.device)
        self.default = torch.tensor([0]).to(self.device)


    def nominal(self, inputs, truth, criterion, model):
        print("#################### du bist in def nominal() #####################")
        return inputs, truth


    def do_not_change(self, inputs, adversarial_vector):
        if self.reduce==False:
            return adversarial_vector

        glob, cpf, npf, vtx = inputs
        vector_glob, vector_cpf, vector_npf, vector_vtx = adversarial_vector

        glob_mask = glob==self.default
        glob_mask[:,self.glob_int] = True
        cpf_mask = cpf==self.default
        cpf_mask[:,:,self.cpf_int] = True
        npf_mask = npf==self.default
        npf_mask[:,:,self.npf_int] = True
        vtx_mask = vtx==self.default
        vtx_mask[:,:,self.vtx_int] = True

        vector_glob = torch.where(glob_mask, self.torch_zero, vector_glob)
        vector_cpf = torch.where(cpf_mask, self.torch_zero, vector_cpf)
        vector_npf = torch.where(npf_mask, self.torch_zero, vector_npf)
        vector_vtx = torch.where(vtx_mask, self.torch_zero, vector_vtx)

        return [vector_glob, vector_cpf, vector_npf, vector_vtx]
    

    def already_fooled(self, adversarial_vector, nominal_labels, adversarial_labels):
        vector_glob, vector_cpf, vector_npf, vector_vtx = adversarial_vector

        mask = (nominal_labels == adversarial_labels).reshape(-1,1)
        glob_mask = mask.expand(-1,15)
        mask = mask.reshape(-1,1,1)
        cpf_mask = mask.expand(-1,25,16)
        npf_mask = mask.expand(-1,25,6)
        vtx_mask = mask.expand(-1,4,12)

        vector_glob = torch.where(glob_mask, vector_glob, self.torch_zero)
        vector_cpf = torch.where(cpf_mask, vector_cpf, self.torch_zero)
        vector_npf = torch.where(npf_mask, vector_npf, self.torch_zero)
        vector_vtx = torch.where(vtx_mask, vector_vtx, self.torch_zero)

        return [vector_glob, vector_cpf, vector_npf, vector_vtx]


    def pgd(self, inputs, truth, criterion, model):
        print("#################### du bist in def pgd() #####################")
        alphas = []
        for e in self.epsilons_per_feature:
            alphas.append(self.epsilon * e / self.iterations)

        adversarial_inputs = []
        for input in inputs:
            adversarial_inputs.append(input.clone().detach().to(self.device).requires_grad_(True))

        for i in range(self.iterations):
            prediction = model(*adversarial_inputs)

            loss = criterion(prediction, truth).mean()

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                gradients = []
                for input in adversarial_inputs:
                    gradients.append(input.grad.detach().sign())

                deltas = []
                for input, alpha, gradient in zip(adversarial_inputs,alphas, gradients):
                    deltas.append(torch.clamp((input + alpha * gradient) - input, min=-alpha * self.iterations, max=alpha * self.iterations))

                deltas = self.do_not_change(inputs, deltas)

                for index, delta in enumerate(deltas):
                    adversarial_inputs[index] += delta

        with torch.no_grad():
            if self.restrict_impact > 0:
                for index, (input, adversarial_input) in enumerate(zip(inputs, adversarial_inputs)):
                    adversarial_input[index] = torch.clamp(adversarial_input[index], min=input - self.restrict_impact * torch.abs(input), max=input + self.restrict_impact * torch.abs(input)).detach()
        print("#################### du bist am ende von def pgd() #####################")
        return adversarial_input, truth


    def jetfool(self, inputs, truth, criterion, model, ):
        print("JetFool attack not yet implemented. Returning nominal inputs.")
        return inputs, truth