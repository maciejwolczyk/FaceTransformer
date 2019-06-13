function indexOfMin(arr) {
    if (arr.length === 0) {
        return -1;
    }

    var min = arr[0];
    var minIndex = 0;

    for (var i = 1; i < arr.length; i++) {
        if (arr[i] < min) {
            minIndex = i;
            min = arr[i];
        }
    }

    return minIndex;
}
let img_width = 96;
let img_height = 96;

resize_image = function() {
    let image= $('#selected-image').get(0);

    let gray = new cv.Mat();
    let faces = new cv.RectVector();

    console.log(face_detector)

    cv_image = cv.imread('selected-image');
    cv.cvtColor(cv_image, gray, cv.COLOR_RGBA2GRAY, 0);

    min_size = [50, 50];
    max_size = [0, 0];
    face_detector.detectMultiScale(gray, faces, 1.2, 10, 0);

    face = faces.get(0);
    max_size = face.width * face.height
    for (let i = 1; i < faces.size(); ++i) {
        current_size = faces.get(i).width * faces.get(i).height
        if(current_size > max_size) {
            max_size = current_size
            face = faces.get(i)
        }
    }
    new_x = Math.floor(face.x - face.width * 0.1);
    new_width = Math.floor(face.width * 1.2);
    new_y = Math.floor(face.y - face.height * 0.1);
    new_height = Math.floor(face.height * 1.2);

    nj_img = nj.images.read(image)
    sliced_image = nj_img.slice([new_y | 0, new_y + new_height | 0], [new_x | 0, new_x + new_width | 0])

    min_dim = Math.min(new_width, new_height);

    center_x = Math.floor(new_width / 2)
    center_y = Math.floor(new_height / 2)
    half_side = Math.floor(min_dim / 2)

    square_image = sliced_image.slice(
        [center_y - half_side, center_y + half_side], 
        [center_x - half_side, center_x + half_side], 
    )
    console.log(square_image.shape)

    // TODO: do this via opencv?
    //
    resized_image = nj.images.resize(square_image, img_height, img_width);
    console.log("resized", resized_image.shape)

    var canvas = document.getElementById('resized-canvas');
    canvas.height = img_height; canvas.width = img_width;
    nj.images.save(resized_image, canvas);
}

document.getElementById("selected-image").onload = function() {
    resize_image()
}

$("#image-selector").change(function(){
    $("#selected-image").attr("src", "");
    let reader = new FileReader();
    
    reader.onload = function(){
        let dataURL = reader.result;
        $("#selected-image").attr("src",dataURL);
        $("#prediction-list").empty();
        console.log("reader onload!")
    }



    

    
    // image_parent = $("#selected-image").parent()
    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
});

let encoder;
let decoder;
let face_detector;

(async function(){
    encoder = await tf.loadGraphModel('tfjs_exports/encoder/model.json');
    decoder = await tf.loadGraphModel('tfjs_exports/decoder/model.json');
    $('.progress-bar').hide();

    let utils = new Utils('errorMessage'); //use utils class
    face_detector = new cv.CascadeClassifier();
    let faceCascadeFile = 'haarcascade_frontalface_default.xml'; // path to xml

    // use createFileFromUrl to "pre-build" the xml
    utils.createFileFromUrl(faceCascadeFile, faceCascadeFile, () => {
        var result = face_detector.load(faceCascadeFile); // in the callback, load the cascade from file
        console.log(result)
    });
})();

transform_face = async function() {
    image = document.getElementById("resized-canvas")
    let tensor = tf.browser.fromPixels(image)
        .toFloat()
        .reshape([-1, img_width * img_height * 3])
        .div(255);
            
    encoded = await encoder.executeAsync(tensor)
    means = encoded[0].arraySync()
    tensor_z = encoded[1]
    console.log("means", means[0], means[3])

    var tensor_means = new Array(4)
    var distance = new Array(4)
    for(i = 0; i < 4; i++) {
        tensor_means[i] = tf.tensor(means[i])
        dist = tf.pow(tf.sub(tensor_means[i], tensor_z), 2)
        distance[i] = tf.sum(dist).dataSync()[0]
    }

    console.log(distance)
    console.log("tensor z", tensor_z, "means", means)

    source_idx = indexOfMin(distance)
    console.log("source idx!", source_idx)

    target_sex = document.getElementById("target-sex").value
    target_smile = document.getElementById("target-smile").value

    target_idx = parseInt(target_sex) * 2 + parseInt(target_smile)

    interpolation_step = parseFloat(document.getElementById("interpolation-step").value) / 100
    console.log(target_idx, interpolation_step) 
    shift = tf.mul(interpolation_step, tf.sub(tensor_means[target_idx], tensor_means[source_idx]))

    shifted_tensor_z = tf.add(encoded[1], shift)
    tensor_image = await decoder.executeAsync(shifted_tensor_z)

    const res = tensor_image.mul(255).toInt().reshape([img_height, img_width,-1])
    const arr = Array.from(res.dataSync())

    var a = nj.array(arr).reshape([img_height, img_width ,3]).tolist()

    var width = img_width,
        height = img_height,
        buffer = new Uint8ClampedArray(width * height * 4);

    for(var y = 0; y < height; y++) {
        for(var x = 0; x < width; x++) {
            var pos = (y * width + x) * 4;
            buffer[pos  ] = a[y][x][0];
            buffer[pos+1] = a[y][x][1];
            buffer[pos+2] = a[y][x][2];
            buffer[pos+3] = 255;
        }
    }

    var canvas = document.getElementById('output-canvas'),
    ctx = canvas.getContext('2d');

    canvas.width = width;
    canvas.height = height;

    var idata = ctx.createImageData(width, height);
    canvas.height = img_height; canvas.width = img_width;

    idata.data.set(buffer);

    ctx.putImageData(idata, 0, 0);
    var dataUri = canvas.toDataURL();
    image.src = dataUri


    arr2 = btoa(JSON.stringify(a))
    console.log(arr2)
    console.log(JSON.stringify(a))

    $('#ItemPreview').attr('src', dataUri);
}

$("#predict-button").click(transform_face);
