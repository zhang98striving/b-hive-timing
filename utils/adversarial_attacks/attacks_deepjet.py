import torch


class Attacks():
    def __init__(self, sample, truth, criterion, device=torch.device("cpu"), epsilon=0.1, epsilon_factors=True, iterations=1 reduce=True, restrict_impact=-1):
        super().__init__()

        self.sample = sample
        self.truth = truth
        self.criterion = criterion
        self.device = device
        self.epsilon = epsilon
        if epsilon_factors:
            print("Individual epsilons per feature not yet implemented. Using epsilon=1 for all variables instead.")
            self.epsilon_glob = 1.0
            self.epsilon_cpf = 1.0
            self.epsilon_npf = 1.0
            self.epsilon_vtx = 1.0
        else:
            self.epsilon_glob = 1.0
            self.epsilon_cpf = 1.0
            self.epsilon_npf = 1.0
            self.epsilon_vtx = 1.0
        self.iterations = iterations
        self.reduce = reduce
        self.restrict_impact = restrict_impact

        self.torch_zero = torch.tensor(0.0).to(self.device)
        self.glob_int = torch.tensor([2, 3, 4, 5, 8, 13, 14]).to(self.device)
        self.cpf_int = torch.tensor([12, 13, 14, 15]).to(self.device)
        self.npf_int = torch.tensor([2]).to(self.device)
        self.vtx_int = torch.tensor([3]).to(self.device)
        self.default = torch.tensor([0]).to(self.device)


    def do_not_change(self, adversarial_vector):
        if self.reduce==False:
            return adversarial_vector

        glob, cpf, npf, vtx = self.sample
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

        return vector_glob, vector_cpf, vector_npf, vector_vtx
    

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

        return vector_glob, vector_cpf, vector_npf, vector_vtx


    def pgd(self, model):
        alpha_glob = self.epsilon * self.epsilon_glob / self.iterations
        alpha_cpf = self.epsilon * self.epsilon_cpf / self.iterations
        alpha_npf = self.epsilon * self.epsilon_npf / self.iterations
        alpha_vtx = self.epsilon * self.epsilon_vtx / self.iterations

        glob, cpf, npf, vtx = self.sample

        adv_glob = glob.clone().detach().to(self.device).requires_grad_(True)
        adv_cpf = cpf.clone().detach().to(self.device).requires_grad_(True)
        adv_npf = npf.clone().detach().to(self.device).requires_grad_(True)
        adv_vtx = vtx.clone().detach().to(self.device).requires_grad_(True)

        for i in range(self.iterations):
            prediction = model(adv_glob, adv_cpf, adv_npf, adv_vtx)

            loss = self.criterion(prediction, self.truth)

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                grad_glob = adv_glob.grad.detach().sign()
                grad_cpf = adv_cpf.grad.detach().sign()
                grad_npf = adv_npf.grad.detach().sign()
                grad_vtx = adv_vtx.grad.detach().sign()

                delta_glob = torch.clamp(
                    adv_glob - (adv_glob + alpha_glob * grad_glob),
                    min=-self.epsilon * self.epsilon_glob,
                    max=self.epsilon * self.epsilon_glob,
                )
                delta_cpf = torch.clamp(
                    adv_cpf - (adv_cpf + alpha_cpf * grad_cpf),
                    min=-self.epsilon * self.epsilon_cpf,
                    max=self.epsilon * self.epsilon_cpf,
                )
                delta_npf = torch.clamp(
                    adv_npf - (adv_npf + alpha_npf * grad_npf),
                    min=-self.epsilon * self.epsilon_npf,
                    max=self.epsilon * self.epsilon_npf,
                )
                delta_vtx = torch.clamp(
                    adv_vtx - (adv_vtx + alpha_vtx * grad_vtx),
                    min=-self.epsilon * self.epsilon_vtx,
                    max=self.epsilon * self.epsilon_vtx,
                )

                delta_glob, delta_cpf, delta_npf, delta_vtx = self.do_not_change((delta_glob, delta_cpf, delta_npf, delta_vtx))

                adv_glob -= delta_glob
                adv_cpf -= delta_cpf
                adv_npf -= delta_npf
                adv_vtx -= delta_vtx

        with torch.no_grad():
            if self.restrict_impact > 0:
                adv_glob = torch.clamp(
                    adv_glob,
                    min=glob - self.restrict_impact * torch.abs(glob),
                    max=glob + self.restrict_impact * torch.abs(glob),
                )
                adv_cpf = torch.clamp(
                    adv_cpf,
                    min=cpf - self.restrict_impact * torch.abs(cpf),
                    max=cpf + self.restrict_impact * torch.abs(cpf),
                )
                adv_npf = torch.clamp(
                    adv_npf,
                    min=npf - self.restrict_impact * torch.abs(npf),
                    max=npf + self.restrict_impact * torch.abs(npf),
                )
                adv_vtx = torch.clamp(
                    adv_vtx,
                    min=vtx - self.restrict_impact * torch.abs(vtx),
                    max=vtx + self.restrict_impact * torch.abs(vtx),
                )

        return adv_glob.detach(), adv_cpf.detach(), adv_npf.detach(), adv_vtx.detach()


    def jetfool(self, model):
        print("JetFool attack not yet implemented. Returning nominal sample.")
        return self.sample