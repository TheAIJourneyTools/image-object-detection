import React, { useState } from "react";

const ImageProcessor = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [processedImage, setProcessedImage] = useState(null);
    const [loading, setLoading] = useState(false);
    const [previewImage, setPreviewImage] = useState(null); // New state for preview image

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setPreviewImage(URL.createObjectURL(file)); // Set preview image URL
    };

    const handleSubmit = async () => {
        setLoading(true);
        if (!selectedFile) {
            alert("Please select an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);

        try {
            const response = await fetch("http://localhost:5000/detect", {
                method: "POST",
                body: formData,
                redirect: "follow",
            });

            if (!response.ok) {
                throw new Error("Failed to process the image");
            }

            // Assuming the backend sends the processed image as a blob
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            setProcessedImage(imageUrl);
            setLoading(false);
        } catch (error) {
            console.error("Error processing the image:", error);
            alert("Error processing the image. Please try again.");
        }
    };

    return (
        <div style={{ padding: "16px", display: "flex", flexDirection: "column", alignItems: "center" }}>
            <h1 style={{ fontSize: "2.25rem", fontWeight: "bold", marginBottom: "16px" }}>Object Detection in Images</h1>
            <div style={{ marginBottom: "16px" }}>
                <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    style={{ display: "flex", marginBottom: "8px", backgroundColor: "#333", padding: "16px", borderRadius: "4px", color: "white" }}
                />
                <button
                    disabled={loading}
                    onClick={handleSubmit}
                    style={{
                        backgroundColor: "#1E90FF",
                        color: "white",
                        padding: "8px 16px",
                        borderRadius: "4px",
                        cursor: "pointer",
                        border: "none",
                        transition: "background-color 0.3s",
                    }}
                    onMouseOver={(e) => e.target.style.backgroundColor = "#1C86EE"}
                    onMouseOut={(e) => e.target.style.backgroundColor = "#1E90FF"}
                >
                    Upload and Process
                </button>
            </div>
            <div>
                {loading && (
                    <p style={{ fontSize: "1.125rem", fontWeight: "bold", marginBottom: "8px" }}>Processing...</p>
                )}
            </div>
            {previewImage && (
                <div style={{ marginTop: "16px", textAlign: "center", padding: "16px", backgroundColor: "#f9f9f9", borderRadius: "4px" }}>
                    <h2 style={{ fontSize: "1.125rem", fontWeight: "bold", marginBottom: "8px" }}>Selected Image:</h2>
                    <img
                        src={previewImage}
                        alt="Selected"
                        style={{ border: "1px solid #ccc", borderRadius: "4px", boxShadow: "0 2px 4px rgba(0,0,0,0.1)" }}
                    />
                </div>
            )}
            {processedImage && (
                <div style={{ marginTop: "16px", textAlign: "center", padding: "16px", backgroundColor: "#f9f9f9", borderRadius: "4px" }}>
                    <h2 style={{ fontSize: "1.125rem", fontWeight: "bold", marginBottom: "8px" }}>Processed Image:</h2>
                    <img
                        src={processedImage}
                        alt="Processed"
                        style={{ border: "1px solid #ccc", borderRadius: "4px", boxShadow: "0 2px 4px rgba(0,0,0,0.1)" }}
                    />
                </div>
            )}
        </div>
    );
};

export default ImageProcessor;
