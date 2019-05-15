var ind = 0;
var jnd = 1;

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

function load(cat, vid) {
    var t_path = "./NCC AMD BEE 10/";
    var vidz = "" + vid;
    while (vidz.length < 5) {
        vidz = "0" + vidz;
    }
    $.ajax({
        method: "GET",
        url: t_path + "dump" + cat + "_" + vidz
    }).done(function (t) {
        $('#dump').html(t);

        t = t.split('\n');

        var l = t[t.length - 1];
        if (l === '') {
            l = t[t.length - 2];
        }
        l = l.split(' ');
        if (l[0] === 'Average') {
            $('#output').append(l[2] + '\n');
        } else {
            $('#output').append('--\n');
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
}
load(videos[ind][0], jnd);
