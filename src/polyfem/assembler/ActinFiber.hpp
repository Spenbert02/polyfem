#pragma once

#include <polyfem/assembler/GenericFiber.hpp>
#include <polyfem/assembler/GenericElastic.hpp>

#include <map>
#include <string>

namespace polyfem::assembler
{
	class ActinFiber : public GenericFiber<ActinFiber>
	{
	public:
		ActinFiber();

		// JSON params:
		//  - "Tmax": scalar / field / param expression (GenericMatParam)
		//  - "activation": scalar / field / param expression in [0,1] (GenericMatParam)
		void add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path) override;

		std::string name() const override { return "ActinFiber"; }
		std::map<std::string, ParamFunc> parameters() const override;

		template <typename T>
		T elastic_energy(
			const RowVectorNd &p,
			const double t,
			const int el_id,
			const DefGradMatrix<T> &def_grad) const
		{
			// compute \Psi = a(t) * T_max * (sqrt(I4) - 1)

			const double Tmax = Tmax_(p, t, el_id);
			const double at = activation_(p, t, el_id);

			const T I4 = I4Bar_generic(p, t, el_id, def_grad, false, false);
			const T lam = sqrt(I4);
			const double a = std::min(1.0, std::max(0.0, at));
			const T Ta = T(Tmax * a);

			// Energy
			return Ta * (lam - T(1));
		}

	private:
		GenericMatParam Tmax_;
		GenericMatParam activation_;
	};
} // namespace polyfem::assembler