module AxDeduplicationCompressor(
  input wire [7:0] uncompressed_data_stream [7:0],
  output wire [7:0] compressed_data
);

  // Parameters
  parameter MAX_ERROR_MAGNITUDE = 4;

  // Internal signals
  reg [7:0] base_element;
  reg [2:0] count;
  reg [7:0] compressed_value;

  // Compression process
  always @* begin
    base_element = uncompressed_data_stream[0];
    count = 1;

    for (int i = 1; i < 8; i = i + 1) begin
      if (abs(uncompressed_data_stream[i] - base_element) <= MAX_ERROR_MAGNITUDE) begin
        count = count + 1;
      end else begin
        compressed_value = {base_element, count};
        base_element = uncompressed_data_stream[i];
        count = 1;
      end
    end

    compressed_value = {base_element, count};
  end

  // Output the compressed data
  assign compressed_data = compressed_value;

endmodule

module AxDeduplicationDecompressor(
  input wire [7:0] compressed_data,
  output wire [7:0] decompressed_data
);

  // Internal signals
  reg [7:0] base_element;
  reg [2:0] count;
  reg [7:0] decompressed_value;

  // Decompression process
  always @* begin
    base_element = compressed_data[7:0];
    count = compressed_data[15:8];

    decompressed_value = 0; // Initialize decompressed value

    // Repeat the base element according to the count
    repeat (count) begin
      decompressed_value = {decompressed_value, base_element};
    end
  end

  // Output the decompressed data
  assign decompressed_data = decompressed_value;

endmodule