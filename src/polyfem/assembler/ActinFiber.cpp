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

	double ActinFiber::cached_activation(const RowVectorNd &p, double t, int el_id) const
	{
		const auto key = std::make_tuple(p(0), p(1), t);

		// python_mutex_ analogue: only guards the map itself, not the Python
		// call below, so a cache hit on one thread never blocks on another
		// thread's in-flight Python evaluation.
		{
			std::lock_guard<std::mutex> lock(activation_mutex_);
			auto it = activation_cache_.find(key);
			if (it != activation_cache_.end())
				return it->second;
		}

		const double res = activation_(p, t, el_id);

		// another thread may have computed and inserted the same key in the
		// meantime; emplace is a no-op in that case and both threads'
		// (identical) results are equivalent.
		{
			std::lock_guard<std::mutex> lock(activation_mutex_);
			activation_cache_.emplace(key, res);
		}
		return res;
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
