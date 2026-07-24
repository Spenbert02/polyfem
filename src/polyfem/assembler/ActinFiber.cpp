#include "ActinFiber.hpp"

namespace polyfem::assembler
{

	ActinFiber::ActinFiber()
		: Tmax_("Tmax"), activation_("activation")
	{
	}

	void ActinFiber::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		GenericFiber::add_multimaterial(index, params, units, root_path);

		Tmax_.add_multimaterial(index, params, units.stress(), root_path);
		activation_.add_multimaterial(index, params, "", root_path);
	}

	std::map<std::string, Assembler::ParamFunc> ActinFiber::parameters() const
	{
		std::map<std::string, ParamFunc> res = GenericFiber<ActinFiber>::parameters();

		const auto &Tmax = this->Tmax_;
		const auto &activation = this->activation_;

		res["Tmax"] = [&Tmax](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return Tmax(p, t, e);
		};

		res["activation"] = [&activation](const RowVectorNd &, const RowVectorNd &p, double t, int e) {
			return activation(p, t, e);
		};

		return res;
	}
} // namespace polyfem::assembler
