import itertools

import numpy as np
import opengm as ogm
import scipy.io as sio

import irsg_core.image_fetch_core as ifc


def generate_pgm(if_data, verbose=False, pred_weight=None):
    # gather data from the if data object
    query_graph = if_data.current_sg_query
    object_detections = if_data.object_detections
    relationship_models = if_data.relationship_models
    image_filename = if_data.image_filename

    # generate the graphical model (vg_data_build_gm_for_image)
    n_objects = len(query_graph.objects)
    n_vars = []
    object_is_detected = []
    query_to_pgm = []
    pgm_to_query = []

    master_box_coords = []

    varcount = 0
    for obj_ix in range(0, n_objects):
        query_object_name = query_graph.objects[obj_ix].names

        # occasionally, there are multiple object names (is 0 the best?)
        if isinstance(query_object_name, np.ndarray):
            query_object_name = query_object_name[0]

        object_name = "obj:" + query_object_name
        if object_name not in object_detections:
            object_is_detected.append(False)
            query_to_pgm.append(-1)
        else:
            if len(master_box_coords) == 0:
                master_box_coords = np.copy(
                    object_detections[object_name][:, 0:4])
            object_is_detected.append(True)
            query_to_pgm.append(varcount)
            varcount += 1
            pgm_to_query.append(obj_ix)

            n_labels = len(object_detections[object_name])
            n_vars.append(n_labels)

    gm = ogm.gm(n_vars, operator='adder')
    if verbose:
        print "number of variables: {0}".format(gm.numberOfVariables)
        for l in range(0, gm.numberOfVariables):
            print "  labels for var {0}: {1}".format(l, gm.numberOfLabels(l))

    functions = []

    # generate 1st order functions for objects
    # TODO: test an uniform dist for missing objects
    if verbose:
        print "unary functions - objects:"

    unary_dets = []
    is_cnn_detected = []
    for obj_ix in range(0, n_objects):
        fid = None

        pgm_ix = query_to_pgm[obj_ix]
        object_name = query_graph.objects[obj_ix].names
        if isinstance(object_name, np.ndarray):
            object_name = object_name[0]
        detail = "unary function for object '{0}'".format(object_name)

        if object_is_detected[obj_ix]:
            if verbose:
                print('  adding {0} as full explicit function '
                      '(qry_ix:{1}, pgm_ix:{2})'.format(
                          detail, obj_ix, pgm_ix))
            is_cnn_detected.append(True)
            prefix_object_name = "obj:" + object_name
            detections = object_detections[prefix_object_name]
            eps = np.finfo(float).eps
            unary_dets.append(detections[:, 4] + eps)
            log_scores = -np.log(detections[:, 4] + eps)
            if pred_weight is not None:
                log_scores *= (1.0 - pred_weight)
            fid = gm.addFunction(log_scores)
        else:
            if verbose:
                print('  skipping {0}, no detection available '
                      '(qry_ix:{1})'.format(object_name, obj_ix))
            continue

        func_detail = ifc.FuncDetail(
            fid, [pgm_ix], "explicit", "object unaries", detail)
        functions.append(func_detail)

    # generate a tracker for storing obj/attr/rel likelihoods (pre-inference)
    tracker = ifc.DetectionTracker(image_filename)
    for i in range(0, n_objects):
        if object_is_detected[i]:
            if isinstance(query_graph.objects[i].names, np.ndarray):
                tracker.object_names.append(query_graph.objects[i].names[0])
            else:
                tracker.object_names.append(query_graph.objects[i].names)
    tracker.unary_detections = unary_dets
    tracker.box_coords = master_box_coords
    tracker.detected_vars = is_cnn_detected

    # generate 2nd order functions for binary relationships
    trip_root = query_graph.binary_triples
    trip_list = []
    if isinstance(trip_root, sio.matlab.mio5_params.mat_struct):
        trip_list.append(query_graph.binary_triples)
    else:
        # if there's only one relationship, we don't have an array :/
        for trip in trip_root:
            trip_list.append(trip)

    # process each binary triple in the list
    if verbose:
        print('binary functions:')
    for trip in trip_list:
        sub_ix = trip.subject
        sub_pgm_ix = query_to_pgm[sub_ix]
        subject_name = query_graph.objects[sub_ix].names
        if isinstance(subject_name, np.ndarray):
            subject_name = subject_name[0]

        obj_ix = trip.object
        obj_pgm_ix = query_to_pgm[obj_ix]
        object_name = query_graph.objects[obj_ix].names
        if isinstance(object_name, np.ndarray):
            object_name = object_name[0]

        relationship = trip.predicate
        bin_trip_key = (subject_name + "_" + relationship.replace(" ", "_") +
                        "_" + object_name)

        # check if there is a gmm for the specific triple string
        if bin_trip_key not in relationship_models:
            if verbose:
                print('  no model for "{0}", generating generic relationship'
                      'string'.format(bin_trip_key))
            bin_trip_key = "*_" + relationship.replace(" ", "_") + "_*"
            if bin_trip_key not in relationship_models:
                if verbose:
                    print('    skipping binary function for relationship'
                          '"{0}", no model available'.format(bin_trip_key))
                continue

        # verify object detections
        if sub_ix == obj_ix:
            if verbose:
                print('    self-relationships not possible in OpenGM, '
                      'skipping relationship')
            continue

        if not object_is_detected[sub_ix]:
            if verbose:
                print('    no detections for object "{0}", skipping '
                      'relationship (qry_ix:{1})'.format(subject_name, sub_ix))
            continue

        if not object_is_detected[obj_ix]:
            if verbose:
                print('    no detections for object "{0}", skipping '
                      'relationship (qry_ix:{1})'.format(object_name, obj_ix))
            continue

        # get model parameters
        prefix_object_name = "obj:" + object_name
        bin_object_box = object_detections[prefix_object_name]

        prefix_subject_name = "obj:" + subject_name
        bin_subject_box = object_detections[prefix_subject_name]

        rel_params = relationship_models[bin_trip_key]

        # generate features from subject and object detection boxes
        cart_prod = np.array(list(itertools.product(
            bin_subject_box, bin_object_box)))
        sub_dim = 0
        obj_dim = 1

        subx_center = cart_prod[:, sub_dim, 0] + 0.5 * cart_prod[:, sub_dim, 2]
        suby_center = cart_prod[:, sub_dim, 1] + 0.5 * cart_prod[:, sub_dim, 3]

        objx_center = cart_prod[:, obj_dim, 0] + 0.5 * cart_prod[:, obj_dim, 2]
        objy_center = cart_prod[:, obj_dim, 1] + 0.5 * cart_prod[:, obj_dim, 3]

        sub_width = cart_prod[:, sub_dim, 2]
        relx_center = (subx_center - objx_center) / sub_width

        sub_height = cart_prod[:, sub_dim, 3]
        rely_center = (suby_center - objy_center) / sub_height

        rel_height = cart_prod[:, obj_dim, 2] / cart_prod[:, sub_dim, 2]
        rel_width = cart_prod[:, obj_dim, 3] / cart_prod[:, sub_dim, 3]

        features = np.vstack(
            (relx_center, rely_center, rel_height, rel_width)).T

        # generate scores => log(epsilon+scores) => platt sigmoid
        scores = ifc.gmm_pdf(features, rel_params.gmm_weights,
                             rel_params.gmm_mu, rel_params.gmm_sigma)
        eps = np.finfo(np.float).eps
        scores = np.log(eps + scores)
        sig_scores = 1.0 / (1. + np.exp(rel_params.platt_a * scores +
                                        rel_params.platt_b))
        log_likelihoods = -np.log(sig_scores)

        tracker.add_group(bin_trip_key, log_likelihoods, bin_object_box,
                          object_name, bin_subject_box, subject_name)

        # generate the matrix of functions
        n_subject_val = len(bin_subject_box)
        n_object_val = len(bin_object_box)
        bin_functions = np.reshape(
            log_likelihoods, (n_subject_val, n_object_val))
        if obj_pgm_ix < sub_pgm_ix:
            bin_functions = bin_functions.T

        # add binary functions to the GM
        detail = "binary functions for relationship '%s'" % (bin_trip_key)
        if verbose:
            print('    adding %s' % detail)
        if pred_weight is not None:
            bin_functions *= pred_weight
        fid = gm.addFunction(bin_functions)

        var_indices = [sub_pgm_ix, obj_pgm_ix]
        if obj_pgm_ix < sub_pgm_ix:
            var_indices = [obj_pgm_ix, sub_pgm_ix]
        func_detail = ifc.FuncDetail(
            fid, var_indices, "explicit", "binary functions", detail)
        functions.append(func_detail)

    # add 1st order factors (fid)
    for f in functions:
        n_indices = len(f.var_indices)
        if n_indices == 1:
            if verbose:
                print('  adding unary factor: {0}'.format(f.detail))
                print('    fid: {0}   var: {1}'.format(
                    f.gm_function_id.getFunctionIndex(), f.var_indices[0]))
            gm.addFactor(f.gm_function_id, f.var_indices[0])
        elif n_indices == 2:
            if verbose:
                print('  adding binary factor: {0}'.format(f.detail))
                print('    fid: {0}   var: {1}'.format(
                    f.gm_function_id.getFunctionIndex(), f.var_indices))
            gm.addFactor(f.gm_function_id, f.var_indices)
        else:
            if verbose:
                print('skipping unexpected factor with {0} '
                      'indices: {1}'.format(n_indices, f.function_type))

    return gm, tracker
