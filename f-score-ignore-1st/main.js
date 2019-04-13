$('#do').click(function () {
    var ntp = 0; //number of true positives
    var nfp = 0; //number of false positives
    var nfn = 0; //number of false negatives

    // get raw data
    var truth = $('#truth').val().split('\n');
    var ground_truth = $('#ground_truth').val().split('\n');
    var data = [];
    for (var i = 1; i < truth.length; i++) {
        data.push(truth[i].split(','));
    }
    truth = data;
    data = [];
    for (var i = 0; i < ground_truth.length; i++) {
        data.push(ground_truth[i].split(' '));
    }
    ground_truth = data;

    // filter
    data = [];
    for (var i = 1; i < ground_truth.length; i++) {
        // gt data
        var gt_frame = JSON.parse(ground_truth[i][0]);
	var gt = {
            ax: JSON.parse(ground_truth[i][1]),
            ay: JSON.parse(ground_truth[i][2]),
            bx: JSON.parse(ground_truth[i][3]),
            by: JSON.parse(ground_truth[i][4]),
            cx: JSON.parse(ground_truth[i][5]),
            cy: JSON.parse(ground_truth[i][6]),
            dx: JSON.parse(ground_truth[i][7]),
            dy: JSON.parse(ground_truth[i][8])
        }

        gt.u = Math.min(gt.ax, gt.bx, gt.cx, gt.dx);
        gt.v = Math.min(gt.ay, gt.by, gt.cy, gt.dy);
        gt.w = Math.max(gt.ax, gt.bx, gt.cx, gt.dx) - gt.u;
        gt.h = Math.max(gt.ay, gt.by, gt.cy, gt.dy) - gt.v;

        // find corresponding t
	var t_index = 0;
        for (var j = 0; j < truth.length; j++) {
            var t_frame = JSON.parse(truth[j][0]);
            if (gt_frame === t_frame) {
                t_index = j;
                break;
            }
        }

        // t data
        var t = {
            u: JSON.parse(truth[t_index][1]),
            v: JSON.parse(truth[t_index][2]),
            w: JSON.parse(truth[t_index][3]),
            h: JSON.parse(truth[t_index][4])
        }

        // intersection
        var x_overlap = Math.max(0, Math.min(t.u + t.w, gt.u + gt.w) - Math.max(t.u, gt.u));
        var y_overlap = Math.max(0, Math.min(t.v + t.h, gt.v + gt.h) - Math.max(t.v, gt.v));
        var intersection = x_overlap * y_overlap;

        var union = (t.w * t.h) + (gt.w * gt.h) - intersection;

        data.push({
            frame: gt_frame,
            t: t,
            gt : gt,
            intersection: intersection,
            union: union,
            iou: intersection / union,
            pascal: (intersection / union) >= 0.5
        });

        if (data[data.length - 1].pascal) {
            ntp++;
        } else {
            nfp++;
        }
    }

    var precision = ntp / (ntp + nfp);
    var recall = ntp / (ntp + nfn);
    var f = (2 * precision * recall) / (precision + recall);

    /*var intersect = [];
    var union = [];
    var pascal = [];
    var start = 1;
    var ntp = 0; //number of true positives
    var nfp = 0; //number of false positives
    var nfn = 0; //number of false negatives
    var t = [];
    var gt = [];
    var initial_w = 0;
    var initial_h = 0;

    start = JSON.parse(ground_truth[0][0]);
    // top bottom left right
    for (var i = 0; i < ground_truth.length - 1; i++) {
	var top = [];
        top[0] = JSON.parse(ground_truth[i][2]);
        top[1] = JSON.parse(ground_truth[i][4]);
        top[2] = JSON.parse(ground_truth[i][6]);
        top[3] = JSON.parse(ground_truth[i][8]);
        var r_top = top[0];
        for (var j = 1; j < 4; j++) {
            if (top[j] < r_top) {
                r_top = top[j];
            }
        }
        //alert(r_top);
        var r_bottom = top[0];
        for (var j = 1; j < 4; j++) {
            if (top[j] > r_top) {
                r_bottom = top[j];
            }
        }
        //alert(r_bottom);
        var ri = [];
        ri[0] = JSON.parse(ground_truth[i][1]);
        ri[1] = JSON.parse(ground_truth[i][3]);
        ri[2] = JSON.parse(ground_truth[i][5]);
        ri[3] = JSON.parse(ground_truth[i][7]);
        var r_right = ri[0];
        for (var j = 1; j < 4; j++) {
            if (ri[j] > r_right) {
                r_right = ri[j];
            }
        }
        //alert(r_right);
        var r_left = ri[0];
        for (var j = 1; j < 4; j++) {
            if (ri[j] < r_left) {
                r_left = ri[j];
            }
        }
        // save gt
        gt.push({
            top: r_top,
            bottom: r_bottom,
            rigth: r_right,
            left: r_left
        });
        if (i === 0) {
            initial_w = gt[0].rigth - gt[0].left;
            initial_h = gt[0].bottom - gt[0].top;
        }
        // get corresponding truth
        var gt_index = JSON.parse(ground_truth[i][0]);
        var t_index = gt_index - start;
        t.push({
            top: JSON.parse(truth[t_index][1 + 1]),
            bottom: JSON.parse(truth[t_index][1 + 1]) + initial_h,
            rigth: JSON.parse(truth[t_index][0 + 1]) + initial_w,
            left: JSON.parse(truth[t_index][0 + 1])
        });
        // intersection
        var d0 = t[i],
                d1 = gt[i],
                d1x = d0.left,
                d1y = d0.top,
                d1xMax = d0.rigth,
                d1yMax = d0.bottom,
                d2x = d1.left,
                d2y = d1.top,
                d2xMax = d1.rigth,
                d2yMax = d1.bottom;
        var x_overlap = Math.max(0, Math.min(d1xMax, d2xMax) - Math.max(d1x, d2x));
        var y_overlap = Math.max(0, Math.min(d1yMax, d2yMax) - Math.max(d1y, d2y));
        intersect.push(x_overlap * y_overlap);
        // union
        union.push((t[i].bottom - t[i].top) * (t[i].rigth - t[i].left)
                + (gt[i].bottom - gt[i].top) * (gt[i].rigth - gt[i].left)
                - intersect[i]);
        // pascal
        pascal.push(intersect[i] / union[i] >= 0.5);
        // f
        if (pascal[i]) {
            ntp++;
        } else {
            nfp++;
        }
    }
    var precision = ntp / (ntp + nfp);
    var recall = ntp / (ntp + nfn);
    var f = (2 * precision * recall) / (precision + recall);
    $('#output').html(JSON.stringify({
        truth: t,
        ground_truth: gt,
        pascal: pascal,
        intersection: intersect,
        union: union,
        f: f,
        ntp: ntp,
        nfp: nfp,
        nfn: nfn
    }));*/
    $('#output').html(JSON.stringify({
        data: data,
        ntp: ntp,
        nfp: nfp,
        nfn: nfn,
        precision: precision,
        recall: recall,
        f: f
    }, null, 4));
});