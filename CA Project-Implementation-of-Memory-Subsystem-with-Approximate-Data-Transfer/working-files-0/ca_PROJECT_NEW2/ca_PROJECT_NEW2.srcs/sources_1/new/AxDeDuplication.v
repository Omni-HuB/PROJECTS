module AxDeduplicationCompressor(
  input wire [7:0] uncompressed_data_stream,
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

    for (integer i = 1; i < 8; i = i + 1) begin
      // Use conditional operator instead of abs function
      if ((uncompressed_data_stream[i] - base_element) <= MAX_ERROR_MAGNITUDE && (uncompressed_data_stream[i] - base_element) >= -MAX_ERROR_MAGNITUDE) begin
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
