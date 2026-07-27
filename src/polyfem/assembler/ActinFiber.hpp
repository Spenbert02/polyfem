#pragma once

#include <polyfem/assembler/GenericFiber.hpp>
#include <polyfem/assembler/GenericElastic.hpp>

#include <map>
#include <mutex>
#include <string>
#include <tuple>

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
			const double at = cached_activation(p, t, el_id);

			const T I4 = I4Bar_generic(p, t, el_id, def_grad, false, true);
			const double a = std::min(1.0, std::max(0.0, at));
			const T Ta = T(Tmax * a);

			// Energy
			return T(0.5) * Ta * (I4 - T(1));
		}

	private:
		// activation_ is Python-backed (see p2_test1_bcs.py's `activation`)
		// and, unlike fiber_direction_ (cached in FiberDirection), was never
		// given a C++-side cache -- every call re-acquired the GIL and
		// re-entered Python. This is a hard-coded, single-parameter version
		// of FiberDirection's existing cache pattern (MatParams.hpp/.cpp).
		// Keyed by (x, y, t) only: activation depends purely on spatial
		// position and time (not which element queries it, and this is a 2D
		// sim so z is always 0), so el_id/z are omitted -- this also lets
		// elements that share a point (e.g. adjacent triangles) share a
		// cache hit instead of each re-entering Python.
		double cached_activation(const RowVectorNd &p, double t, int el_id) const;

		GenericMatParam Tmax_;
		GenericMatParam activation_;

		mutable std::mutex activation_mutex_;
		mutable std::map<std::tuple<double, double, double>, double> activation_cache_;
	};
} // namespace polyfem::assembler