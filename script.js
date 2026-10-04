const form =
    document.getElementById("predictionForm");


form.addEventListener(
    "submit",
    function () {

        const button =
            form.querySelector("button");

        button.innerText =
            "Predicting...";

        button.disabled = true;

    }
);