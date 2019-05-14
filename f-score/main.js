var ind = 0;
var jnd = 1;

function load(cat, vid) {
    var gt_path = "./alov300++GT_txtFiles/alov300++_rectangleAnnotation_full/";
    var t_path = "./res/NCC AMD BEE 4/";
    var vidz = "" + vid;
    while (vidz.length < 5) {
        vidz = "0" + vidz;
    }
    $.ajax({
        method: "GET",
        url: gt_path + cat + "/" + cat + "_video" + vidz + ".ann"
    }).done(function (t) {
        $('#ground_truth').html(t);
        $.ajax({
            method: "GET",
            url: t_path + cat + "_" + vid
        }).done(function (t) {
            $('#truth').html(t);
            $('#do').click();
        });
    });
}

// all videos
var videos = [
    ["01-Light", 33],
    ["02-SurfaceCover", 15],
    ["03-Specularity", 18],
    ["04-Transparency", 20],
    ["05-Shape", 24],
    ["06-MotionSmoothness", 22],
    ["07-MotionCoherence", 12],
    ["08-Clutter", 15],
    ["09-Confusion", 37],
    ["10-LowContrast", 23],
    ["11-Occlusion", 34],
    ["12-MovingCamera", 22],
    ["13-ZoomingCamera", 29],
    ["14-LongDuration", 10]
];
load(videos[ind][0], jnd);

$('#do').click(function () {
    var video_fail = 0;

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
    for (var i = 0; i < ground_truth.length; i++) {
        // gt data
        if (ground_truth[i][0] === '') {
            break;
        }
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
        };

        gt.u = Math.min(gt.ax, gt.bx, gt.cx, gt.dx);
        gt.v = Math.min(gt.ay, gt.by, gt.cy, gt.dy);
        gt.w = Math.max(gt.ax, gt.bx, gt.cx, gt.dx) - gt.u;
        gt.h = Math.max(gt.ay, gt.by, gt.cy, gt.dy) - gt.v;

        // find corresponding t
        var t_index = 0;
        for (var j = 0; j < truth.length; j++) {
            if (truth[j][0] === '') {
                video_fail = 1;
                break;
            }
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
        };

        // intersection
        var x_overlap = Math.max(0, Math.min(t.u + t.w, gt.u + gt.w) - Math.max(t.u, gt.u));
        var y_overlap = Math.max(0, Math.min(t.v + t.h, gt.v + gt.h) - Math.max(t.v, gt.v));
        var intersection = x_overlap * y_overlap;

        var union = (t.w * t.h) + (gt.w * gt.h) - intersection;

        data.push({
            frame: gt_frame,
            t: t,
            gt: gt,
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

    $('#output').html(JSON.stringify({
        data: data,
        ntp: ntp,
        nfp: nfp,
        nfn: nfn,
        precision: precision,
        recall: recall,
        f: f
    }, null, 4));
    if (video_fail === 0) {
        $('#output2').append(f + '\n');
    } else {
        $('#output2').append('--\n');
    }

    // next video
    jnd++;
    if (jnd > videos[ind][1]) {
        jnd = 1;
        ind++;
    }
    if (ind < videos.length) {
        load(videos[ind][0], jnd);
    }
});
