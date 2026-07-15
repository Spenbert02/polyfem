#include "BoundaryMeasureLagrangianForm.hpp"

namespace polyfem::solver
{
	/// NOTE: assumes P1 Lagrange simplicial basis functions
	/// NOTE: assumes isoparametric (ie, bases == geom_bases)
	/// NOTE: does not support remeshing
	BoundaryMeasureLagrangianForm::BoundaryMeasureLagrangianForm(
		const int ndof,
		const mesh::Mesh &mesh,
		const std::vector<mesh::LocalBoundary> &local_boundary_measure,
		const std::vector<basis::ElementBases> &bases,
		const std::vector<basis::ElementBases> &geom_bases,
		const assembler::Problem &problem,
		const int boundary_id) : dim_(mesh.dimension()), n_dof_(ndof), problem_(problem), boundary_id_(boundary_id)
	{
		if (dim_ != 2)
			log_and_throw_error("3D not yet supported for boundary_measure");

		// collect data
		for (const auto &lb : local_boundary_measure)
		{
			assert(lb.size() == 1);
			const auto &b = bases[lb.element_id()];
			const auto local_nodes = b.local_nodes_for_primitive(lb.global_primitive_id(0), mesh);
			assert(local_nodes.size() == dim_);

			// collect node global indices
			Eigen::VectorXi primitive(dim_);
			for (int n = 0; n < local_nodes.size(); n++)
			{
				const auto &basis = b.bases[local_nodes(n)];
				assert(basis.global().size() == 1);   // assume P1 lagrange elements
				assert(basis.global()[0].val == 1.0); // NOTE: not really required if above passes, just out of curiosity
				const int global_id = basis.global()[0].index;
				const RowVectorNd rest_pos = basis.global()[0].node;
				if (global2local_node_indices_.count(global_id) == 0)
				{
					rest_positions_.emplace_back(rest_pos);
					int local_idx = global2local_node_indices_.size();
					global2local_node_indices_[global_id] = local_idx;
					local2global_node_indices_[local_idx] = global_id;
				}
				primitive(n) = global_id;
			}
			global_connectivity_.push_back(primitive);
		}

		// ensure single, manifold, closed boundary
		if (dim_ == 2)
		{
			if (!is_single_closed_manifold_boundary_curve())
				log_and_throw_error("Boundary measure id {} is not a single closed manifold curve", boundary_id_);
		}
		else if (dim_ == 3)
		{
			if (!is_single_closed_manifold_boundary_surface())
				log_and_throw_error("Boundary measure id {} is not a single closed manifold surface", boundary_id_);
		}
		else
		{
			log_and_throw_error("What the hell is dim: {}", dim_);
		}

		// compute initial length / area
		L0_ = (dim_ == 2) ? compute_length(rest_positions_, global_connectivity_) : compute_area(rest_positions_, global_connectivity_);
	}

	bool BoundaryMeasureLagrangianForm::is_single_closed_manifold_boundary_curve() const
	{
		if (dim_ != 2)
			log_and_throw_error("This function is for 2d only");

		// precompute adjacency matrix
		std::map<int, std::set<int>> adj;
		for (int i = 0; i < global_connectivity_.size(); i++)
		{
			adj[global_connectivity_[i](0)].insert(global_connectivity_[i](1));
			adj[global_connectivity_[i](1)].insert(global_connectivity_[i](0));
		}

		// run BFS, check manifoldness at each node
		auto [g_idx, l_idx] = *global2local_node_indices_.begin();
		std::set<int> visited;
		visited.insert(g_idx);
		std::queue<int> q;
		q.push(g_idx);
		while (!q.empty())
		{
			const int curr_g_idx = q.front();
			q.pop();
			const auto curr_adj = adj[curr_g_idx];
			if (curr_adj.size() != 2)
				log_and_throw_error("Boundary curve is not manifold");
			for (const int nb_g_idx : curr_adj)
			{
				if (visited.count(nb_g_idx) == 0)
				{ // unvisited node
					visited.insert(nb_g_idx);
					q.push(nb_g_idx);
				}
			}
		}

		return (visited.size() == global2local_node_indices_.size());
	}

	bool BoundaryMeasureLagrangianForm::is_single_closed_manifold_boundary_surface() const
	{
		log_and_throw_error("Boundary check not yet implemented for 3d.");
	}

	double BoundaryMeasureLagrangianForm::compute_length(
		const std::vector<Eigen::VectorXd> &V,
		const std::vector<Eigen::VectorXi> &E_glob) const
	{
		if (dim_ != 2)
			log_and_throw_error("Dimension is {} != 2 for computing length.", dim_);

		double ret = 0.0;
		for (int i = 0; i < E_glob.size(); i++)
		{
			const int loc_idx0 = global2local_node_indices_.at(E_glob[i](0));
			const int loc_idx1 = global2local_node_indices_.at(E_glob[i](1));
			ret += (V[loc_idx0] - V[loc_idx1]).norm();
		}
		return ret;
	}

	double BoundaryMeasureLagrangianForm::compute_area(
		const std::vector<Eigen::VectorXd> &V,
		const std::vector<Eigen::VectorXi> &F_glob) const
	{
		log_and_throw_error("compute_area not yet implemented.");
	}

	double BoundaryMeasureLagrangianForm::value_unweighted(const Eigen::VectorXd &x) const
	{
		const double L = compute_measure_from_displacements(x);
		const double L_penalty = -lagr_mult_ * (L - L_target_);
		const double A_penalty = 0.5 * (L - L_target_) * (L - L_target_);
		return L_weight() * L_penalty + A_weight() * A_penalty;
	}

	// compute S * (grad_x L), with S = (-lam + mu * (L - L*))
	void BoundaryMeasureLagrangianForm::first_derivative_unweighted(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		compute_gradL(x, gradv);
		const double L = compute_measure_from_displacements(x);
		gradv = ((-L_weight() * lagr_mult_) + (A_weight() * (L - L_target_))) * gradv;
	}

	// compute H = A * (grad_x L) * (grad_x L)^T  +  S * (grad2_x L), with S = (-lam + mu * (L - L*))
	void BoundaryMeasureLagrangianForm::second_derivative_unweighted(const Eigen::VectorXd &x, StiffnessMatrix &hessian) const
	{
		Eigen::VectorXd gradL;
		compute_gradL(x, gradL);

		const int size = x.size();
		hessian.resize(size, size);

		// gradL is only nonzero at this boundary's own nodes, so restrict the dense
		// outer product to those global dofs instead of the entire (potentially huge)
		// global dof vector -- inserting a full size x size dense block here is both
		// a severe perf problem and a numerical one (CHOLMOD chokes on the resulting
		// near-dense-but-mostly-zero matrix).
		std::vector<int> dof_indices;
		dof_indices.reserve(local2global_node_indices_.size() * dim_);
		for (const auto &[local_idx, global_idx] : local2global_node_indices_)
			for (int d = 0; d < dim_; d++)
				dof_indices.push_back(global_idx * dim_ + d);

		std::vector<Eigen::Triplet<double>> triplets;
		triplets.reserve(dof_indices.size() * dof_indices.size() + global_connectivity_.size() * 4 * dim_ * dim_);

		// populate dense outer product (restricted to this boundary's dofs)
		for (const int gj : dof_indices)
			for (const int gi : dof_indices)
				triplets.emplace_back(gi, gj, gradL(gi) * gradL(gj) * A_weight());

		// sparse part
		const double S = (-L_weight() * lagr_mult_) + (A_weight() * (compute_measure_from_displacements(x) - L_target_));
		auto add_block = [&](int bi, int bj, const Eigen::MatrixXd &block) {
			int g_i_start = bi * dim_;
			int g_j_start = bj * dim_;
			for (int lj = 0; lj < dim_; lj++)
			{
				for (int li = 0; li < dim_; li++)
				{
					int g_i = g_i_start + li;
					int g_j = g_j_start + lj;
					triplets.emplace_back(g_i, g_j, block(li, lj));
				}
			}
		};
		for (const Eigen::VectorXi &edge : global_connectivity_)
		{
			const int g_idx_1 = edge(0);
			const int g_idx_2 = edge(1);
			const Eigen::VectorXd p1 = x.segment(g_idx_1 * dim_, dim_) + rest_positions_[global2local_node_indices_.at(g_idx_1)];
			const Eigen::VectorXd p2 = x.segment(g_idx_2 * dim_, dim_) + rest_positions_[global2local_node_indices_.at(g_idx_2)];
			const double l = (p2 - p1).norm();
			const Eigen::VectorXd e_hat = (p2 - p1) / l;
			const Eigen::MatrixXd P = S * (1 / l) * (Eigen::MatrixXd::Identity(dim_, dim_) - (e_hat * e_hat.transpose()));
			add_block(g_idx_1, g_idx_1, P);
			add_block(g_idx_2, g_idx_2, P);
			add_block(g_idx_1, g_idx_2, -1.0 * P);
			add_block(g_idx_2, g_idx_1, -1.0 * P);
		}

		hessian.setFromTriplets(triplets.begin(), triplets.end());
	}

	// NOTE: need to check for time dependence?
	void BoundaryMeasureLagrangianForm::update_quantities(const double t, const Eigen::VectorXd &x)
	{
		update_target(t);
	}

	void BoundaryMeasureLagrangianForm::update_lagrangian(const Eigen::VectorXd &x, const double k_al)
	{
		const double L = compute_measure_from_displacements(x);
		k_al_ = k_al;
		lagr_mult_ = k_al * (L - L_target_);
	}

	double BoundaryMeasureLagrangianForm::compute_error(const Eigen::VectorXd &x) const
	{
		const double L = compute_measure_from_displacements(x);
		return (L - L_target_) * (L - L_target_);
	}

	double BoundaryMeasureLagrangianForm::compute_measure_from_displacements(const Eigen::VectorXd &x) const
	{
		// extract deformed positions, compute L
		std::vector<Eigen::VectorXd> deformed_positions;
		deformed_positions.reserve(rest_positions_.size());
		for (int i = 0; i < rest_positions_.size(); i++)
		{
			const int g_idx = local2global_node_indices_.at(i);
			Eigen::VectorXd u = x.segment(g_idx * dim_, dim_); // displacement
			deformed_positions.emplace_back(rest_positions_[i] + u);
		}
		const double L = (dim_ == 2)
							 ? compute_length(deformed_positions, global_connectivity_)
							 : compute_area(deformed_positions, global_connectivity_);
		return L;
	}

	void BoundaryMeasureLagrangianForm::compute_gradL(const Eigen::VectorXd &x, Eigen::VectorXd &gradv) const
	{
		gradv.resizeLike(x);
		gradv.setZero();

		for (int i = 0; i < rest_positions_.size(); i++)
		{
			const int g_idx = local2global_node_indices_.at(i);
			const int l_idx_prev = i == 0 ? rest_positions_.size() - 1 : i - 1;
			const int g_idx_prev = local2global_node_indices_.at(l_idx_prev);
			const int l_idx_next = i == rest_positions_.size() - 1 ? 0 : i + 1;
			const int g_idx_next = local2global_node_indices_.at(l_idx_next);
			const Eigen::VectorXd p = rest_positions_[i] + x.segment(g_idx * dim_, dim_);
			const Eigen::VectorXd p_prev = rest_positions_[l_idx_prev] + x.segment(g_idx_prev * dim_, dim_);
			const Eigen::VectorXd p_next = rest_positions_[l_idx_next] + x.segment(g_idx_next * dim_, dim_);
			gradv.segment(g_idx * dim_, dim_) = (p - p_prev).normalized() + (p - p_next).normalized();
		}
	}
} // namespace polyfem::solver