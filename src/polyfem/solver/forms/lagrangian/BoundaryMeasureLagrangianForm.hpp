#pragma once

#include "AugmentedLagrangianForm.hpp"
#include <polyfem/mesh/LocalBoundary.hpp>
#include <polyfem/assembler/Problem.hpp>

namespace polyfem::solver
{
	/// @brief Form of the augmented lagrangian for constraining a surface measure
	class BoundaryMeasureLagrangianForm : public AugmentedLagrangianForm
	{
	public:
		BoundaryMeasureLagrangianForm(
			const int ndof,
			const mesh::Mesh &mesh,
			const std::vector<mesh::LocalBoundary> &local_boundary_measure,
			const std::vector<basis::ElementBases> &bases,
			const std::vector<basis::ElementBases> &geom_bases,
			const assembler::Problem &problem,
			const int boundary_id);

		std::string name() const override
		{
			return "boundary-measure-alagrangian";
		}

		/// @brief Compute the value of the form
		/// @param x Current solution
		/// @return Computed value
		double value_unweighted(const Eigen::VectorXd &x) const override;

		/// @brief Compute the first derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] gradv Output gradient of the value wrt x
		void first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const override;

		/// @brief Compute the second derivative of the value wrt x
		/// @param[in] x Current solution
		/// @param[out] hessian Output hessian of the value wrt
		void second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const override;

	public:
		/// @brief Update time dependent quantities
		/// @param t New time
		/// @param x Solution at time t
		void update_quantities(const double t, const Eigen::VectorXd &x) override;

		void update_lagrangian(const Eigen::VectorXd &x, const double k_al) override;

		double compute_error(const Eigen::VectorXd &x) const override;

	private:
		const int dim_;
		const int n_dof_;
		const assembler::Problem &problem_;
		const int boundary_id_;
		double L0_;       // rest configuration
		double L_target_; // target, computed at each t
		double lagr_mult_ = 0;
		std::map<int, int> global2local_node_indices_; // global2local_node_indices[glob_ind] = loc_ind
		std::map<int, int> local2global_node_indices_; // local2global_node_indices[loc_ind] = glob_ind
		std::vector<Eigen::VectorXd> rest_positions_;
		// std::map<int, std::set<int>> global_connectivity_;
		std::vector<Eigen::VectorXi> global_connectivity_;

		/// @brief update target L to the prescribed value at time t
		/// @param t
		void update_target(const double t)
		{
			L_target_ = L0_ + problem_.boundary_measure_bc(boundary_id_, t);
			if (L_target_ < 0)
				log_and_throw_error(
					"Negative target boundary measure {}-{}={} computed at t={}",
					L0_, problem_.boundary_measure_bc(boundary_id_, t), L_target_, t);
		}

		/// @brief check if boundary is closed and manifold
		bool is_single_closed_manifold_boundary_curve() const;
		bool is_single_closed_manifold_boundary_surface() const;

		double compute_length(const std::vector<Eigen::VectorXd> &V, const std::vector<Eigen::VectorXi> &E_glob) const;
		double compute_area(const std::vector<Eigen::VectorXd> &V, const std::vector<Eigen::VectorXi> &F_glob) const;
		double compute_measure_from_displacements(const Eigen::VectorXd &x) const;
		void compute_gradL(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const;
	};

} // namespace polyfem::solver