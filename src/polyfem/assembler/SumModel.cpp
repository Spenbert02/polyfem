#include "SumModel.hpp"

#include <jse/jse.h>

// #include <polyfem/basis/Basis.hpp>
// #include <polyfem/autogen/auto_elasticity_rhs.hpp>

// #include <igl/Timer.h>

namespace polyfem::assembler
{
	void SumModel::add_multimaterial(const int index, const json &params, const Units &units, const std::string &root_path)
	{
		assert(size() == 2 || size() == 3);
		if (params.count("models") == 0)
			return;

		auto models = params["models"];
		element_types_[index] = std::vector<std::string>();

		for (const auto &model : models)
		{
			const std::string model_name = model["type"];
			if (assemblers_by_type_.find(model_name) == assemblers_by_type_.end())
			{
				const auto assembler = AssemblerUtils::make_assembler(model_name);
				assemblers_by_type_[model_name] = std::dynamic_pointer_cast<NLAssembler>(assembler);
				assert(assemblers_by_type_[model_name] != nullptr);
				assemblers_by_type_[model_name]->set_size(size());
			}
			assemblers_by_type_[model_name]->add_multimaterial(index, model, units, root_path);
			element_types_[index].push_back(model_name);
		}
	}

	Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1>
	SumModel::compute_rhs(const AutodiffHessianPt &pt) const
	{
		assert(pt.size() == size());
		Eigen::Matrix<double, Eigen::Dynamic, 1, 0, 3, 1> res;
		assert(false);

		return res;
	}

	Eigen::VectorXd
	SumModel::assemble_gradient(const NonLinearAssemblerData &data) const
	{
		const auto &model_names = element_types_.at(data.vals.element_id);
		Eigen::VectorXd gradient = assemblers_by_type_.at(model_names[0])->assemble_gradient(data);
		for (size_t i = 1; i < model_names.size(); i++)
		{
			gradient += assemblers_by_type_.at(model_names[i])->assemble_gradient(data);
		}
		return gradient;
	}

	Eigen::MatrixXd
	SumModel::assemble_hessian(const NonLinearAssemblerData &data) const
	{
		const auto &model_names = element_types_.at(data.vals.element_id);
		Eigen::MatrixXd hessian = assemblers_by_type_.at(model_names[0])->assemble_hessian(data);
		for (size_t i = 1; i < model_names.size(); i++)
		{
			hessian += assemblers_by_type_.at(model_names[i])->assemble_hessian(data);
		}
		return hessian;
	}

	double SumModel::compute_energy(const NonLinearAssemblerData &data) const
	{
		const auto &model_names = element_types_.at(data.vals.element_id);
		double energy = assemblers_by_type_.at(model_names[0])->compute_energy(data);
		for (size_t i = 1; i < model_names.size(); i++)
		{
			energy += assemblers_by_type_.at(model_names[i])->compute_energy(data);
		}
		return energy;
	}

	void SumModel::assign_stress_tensor(
		const OutputData &data,
		const int all_size,
		const ElasticityTensorType &type,
		Eigen::MatrixXd &all,
		const std::function<Eigen::MatrixXd(const Eigen::MatrixXd &)> &fun) const
	{
		all.resize(data.local_pts.rows(), all_size);
		all.setZero();

		Eigen::MatrixXd tmp;
		const auto &model_names = element_types_.at(data.el_id);

		if (type == ElasticityTensorType::F)
		{
			std::dynamic_pointer_cast<assembler::ElasticityAssembler>(assemblers_by_type_.at(model_names[0]))
				->assign_stress_tensor(data, all_size, type, all, fun);
			return;
		}

		for (const auto &model : model_names)
		{
			const auto &assembler = assemblers_by_type_.at(model);
			std::dynamic_pointer_cast<assembler::ElasticityNLAssembler>(assembler)->assign_stress_tensor(data, all_size, type, tmp, fun);
			all += tmp;
		}
	}

	std::map<std::string, Assembler::ParamFunc> SumModel::parameters() const
	{
		std::map<std::string, Assembler::ParamFunc> params;
		for (const auto &[model_name, a] : assemblers_by_type_)
		{
			for (const auto &[param_name, func] : a->parameters())
			{
				const std::string full_name = model_name + "/" + param_name;
				params[full_name] = [this, model_name, func](const RowVectorNd &uv, const RowVectorNd &p, double t, int e) {
					const auto it = element_types_.find(e);
					if (it == element_types_.end() || std::find(it->second.begin(), it->second.end(), model_name) == it->second.end())
						return 0.0;
					return func(uv, p, t, e);
				};
			}
		}
		return params;
	}
} // namespace polyfem::assembler
